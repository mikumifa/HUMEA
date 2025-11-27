import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn

from layers import MoEAdaptorLayer, MultiHeadGraphAttention


class GAT(nn.Module):
    def __init__(
        self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag
    ):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                MultiHeadGraphAttention(
                    n_heads[i],
                    f_in,
                    n_units[i + 1],
                    attn_dropout,
                    diag,
                    nn.init.ones_,
                    False,
                )
            )

    def forward(self, x, adj, g_device):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj, g_device)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x


def list_rebul_sort(list, emb, k):
    def simility_cal(emb):
        sim_matric = cosine_similarity(emb, emb)
        return sim_matric

    sim_m = simility_cal(emb.cpu().detach().numpy())
    new_list = []  # type: ignore
    ns_list = []
    for i in range(0, list.shape[0]):
        count = 0
        s_b = sim_m[i].argsort()
        # print(s_b)
        ns_list.append(list[i])
        for j in s_b[::-1]:
            if j == i:
                continue
            if j + 1 in np.array(ns_list):
                continue
            else:
                if j + 1 in list:
                    ns_list.append(list[np.where(list == (j + 1))[0][0]])
                    count = count + 1
                    if count == k:
                        # print('enough')
                        break

        new_list = new_list + ns_list
        ns_list.clear()

    new_list = np.array(new_list)
    return new_list


class CLUBSample(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-((mu - y_samples) ** 2) / 2.0 / logvar.exp()).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = -((mu - y_samples) ** 2) / logvar.exp()
        negative = -((mu - y_samples[random_index]) ** 2) / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.0

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


class MIEstimator(nn.Module):
    def __init__(self, args):
        super(MIEstimator, self).__init__()
        self.num = args.n_exp

        self.estimators = nn.ModuleDict(
            {
                "img": CLUBSample(args.feat_dim, args.feat_dim, args.feat_dim),
                "rel": CLUBSample(args.feat_dim, args.feat_dim, args.feat_dim),
                "rel_text": CLUBSample(args.feat_dim, args.feat_dim, args.feat_dim),
                "att": CLUBSample(args.feat_dim, args.feat_dim, args.feat_dim),
                "att_text": CLUBSample(args.feat_dim, args.feat_dim, args.feat_dim),
            }
        )

    def forward(self, embeddings: dict):
        bsz, n_exp, _ = embeddings["img"].size()
        assert n_exp == self.num
        mi_losses = []
        for key in embeddings.keys():
            idx1, idx2 = random.sample(range(n_exp), 2)
            z1 = embeddings[key][:, idx1, :]
            z2 = embeddings[key][:, idx2, :]
            mi_losses.append(self.estimators[key](z1, z2))

        return sum(mi_losses) / len(mi_losses)

    def train_estimator(self, embeddings: dict):
        bsz, n_exp, _ = embeddings["img"].size()
        assert n_exp == self.num

        est_losses = []

        for key in embeddings.keys():
            idx1, idx2 = random.sample(range(n_exp), 2)
            z1 = embeddings[key][:, idx1, :]
            z2 = embeddings[key][:, idx2, :]
            est_losses.append(self.estimators[key].learning_loss(z1, z2))

        return sum(est_losses) / len(est_losses)


class PowerSetFusionWithMoEGating(nn.Module):
    def __init__(
        self,
        args,
        num_modalities=6,
        combos=["010000", "000100", "000001", "001010", "101000", "100010", "101010"],
    ):
        super().__init__()
        self.num_combos = len(combos)
        if args.fusion_weight_dim == 0:
            self.use_proj = False
            weight = torch.zeros(self.num_combos, 1)
            logger.info(f"weight :{weight}")
            self.weight = nn.Parameter(weight, requires_grad=True)
        else:
            self.use_proj = True
            hidden_dim = args.fusion_weight_dim
            self.combo_gate_query = nn.Parameter(torch.randn(hidden_dim))
            self.combo_proj = nn.Linear(hidden_dim, self.num_combos)
        self.num_modalities = num_modalities
        self.all_combo_masks = self.build_combo_masks(num_modalities)  # [6, 63]
        if combos:
            self.combos = combos
            self.combo_indices = self.select_combo_indices(combos)
        else:
            self.combo_indices = list(range(2**num_modalities - 1))
        logger.info(f"combo_indices :{self.combo_indices}")
        self.topk = args.topk

        self.register_buffer("combo_masks", self.all_combo_masks[:, self.combo_indices])

    def build_combo_masks(self, n):
        masks = []
        for i in range(n):
            mask = []
            for j in range(1, 2**n):
                bitmask = format(j, f"0{n}b")
                mask.append(int(bitmask[n - 1 - i]))
            masks.append(torch.tensor(mask).float())
        return torch.stack(masks)  # [6, 63]

    def select_combo_indices(self, combos):
        """
        Convert binary combo strings to indices [1..63]
        """
        indices = []
        for combo in combos:
            combo = combo[::-1]
            idx = int(combo, 2)  # binary string -> int
            if idx == 0 or idx >= 2**self.num_modalities:
                raise ValueError(f"Invalid combo: {combo}")
            indices.append(idx - 1)  # offset by 1, since mask range is [1..63]
        return indices

    def forward(self, inputs):
        assert len(inputs) == self.num_modalities
        if self.use_proj:
            weight_norm = F.softmax(self.combo_proj(self.combo_gate_query), dim=0)
        else:
            weight_norm = F.softmax(self.weight, dim=0)
        _, topk_indices = torch.topk(weight_norm.squeeze(), self.topk)
        mask = torch.zeros_like(weight_norm)
        mask[topk_indices] = 1.0
        masked_weight = weight_norm * mask

        weight_norm = masked_weight / masked_weight.sum()

        # [num_combos] @ [num_combos, 6] => [6]
        modal_weights = torch.matmul(weight_norm.squeeze(-1), self.combo_masks.T)  # [6]
        if not self.training:
            self.latest_weights = modal_weights.detach()
        embs = [F.normalize(x) * modal_weights[i] for i, x in enumerate(inputs)]
        fused = torch.cat(embs, dim=-1)  # [B, 6*D]

        return fused


class MultiModalEncoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(
        self,
        args,
        ent_num,
    ):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        self.ENT_NUM = ent_num

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True
        self.joint = PowerSetFusionWithMoEGating(args)
        self.gph_emb = GAT(
            n_units=self.n_units,
            n_heads=self.n_heads,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            instance_normalization=self.args.instance_normalization,
            diag=True,
        )
        self.img_fc = MoEAdaptorLayer(n_exps=args.n_exp, layers=[4096, args.feat_dim])
        self.att_txt_fc = MoEAdaptorLayer(
            n_exps=args.n_exp, layers=[768, args.feat_dim]
        )
        self.rel_txt_fc = MoEAdaptorLayer(
            n_exps=args.n_exp, layers=[768, args.feat_dim]
        )
        self.att_fc = MoEAdaptorLayer(n_exps=args.n_exp, layers=[1000, args.feat_dim])
        self.rel_fc = MoEAdaptorLayer(n_exps=args.n_exp, layers=[1000, args.feat_dim])

    def forward(
        self,
        e_device,
        input_idx,
        adj,
        img_features,
        rel_features,
        att_features,
        att_text_features,
        rel_text_features,
        exp_outputs=False,
    ):
        gph_emb = self.gph_emb(self.entity_emb(input_idx).to(e_device), adj, e_device)
        img_emb, img_outputs, _ = self.img_fc(img_features)
        rel_emb, rel_outputs, _ = self.rel_fc(rel_features)
        rel_text_emb, rel_text_outputs, _ = self.rel_txt_fc(rel_text_features)
        att_emb, att_outputs, _ = self.att_fc(att_features)
        att_text_emb, att_text_outputs, _ = self.att_txt_fc(att_text_features)
        modalities = {
            1: "img_emb",
            2: "gph_emb",
            3: "rel_emb",
            4: "rel_text_emb",
            5: "att_emb",
            6: "att_text_emb",
        }

        # Ablation
        if self.args.without in modalities:
            name = modalities[self.args.without]
            target = locals()[name]
            zero_filled = torch.zeros_like(target)
            if name == "img_emb":
                img_emb = zero_filled
            elif name == "gph_emb":
                gph_emb = zero_filled
            elif name == "rel_emb":
                rel_emb = zero_filled
            elif name == "rel_text_emb":
                rel_text_emb = zero_filled
            elif name == "att_emb":
                att_emb = zero_filled
            elif name == "att_text_emb":
                att_text_emb = zero_filled

        joint_emb = self.joint(
            [gph_emb, img_emb, rel_emb, rel_text_emb, att_emb, att_text_emb]
        )
        # --- Output ---
        if exp_outputs:
            return (
                [
                    gph_emb,
                    img_emb,
                    rel_emb,
                    att_emb,
                    att_text_emb,
                    rel_text_emb,
                    joint_emb,
                ],
                {
                    "img": img_outputs,
                    "rel": rel_outputs,
                    "rel_text": rel_text_outputs,
                    "att": att_outputs,
                    "att_text": att_text_outputs,
                },
            )

        return (
            gph_emb,
            img_emb,
            rel_emb,
            att_emb,
            att_text_emb,
            rel_text_emb,
            joint_emb,
        )
