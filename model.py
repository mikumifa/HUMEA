import math
from typing import Optional
from loguru import logger
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
import torch
import torch.nn.functional as F
from layers import GraphConvolution, MoEAdaptorLayer, MultiHeadGraphAttention


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


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs"""
    return im.mm(s.t())


def l2norm(X):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X) + 1e-8
    X = torch.div(X, a)
    return X


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(
            torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad
        )

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [
            weight_norm[idx] * F.normalize(embs[idx])
            for idx in range(self.modal_num)
            if embs[idx] is not None
        ]
        joint_emb = torch.cat(embs, dim=1)
        # joint_emb = torch.sum(torch.stack(embs, dim=1), dim=1)
        return joint_emb


class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
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


class MultiHeadAttention(nn.Module):
    def __init__(
        self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True
    ):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)

        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.matmul(query, key.transpose(-1, -2))

    def forward(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        seq_len, _ = query.shape  # 27793

        query = self.query(query)  # torch.Size([27793, 10, 10])
        key = self.key(key)  # torch.Size([27793, 10, 10])
        value = self.value(value)  # torch.Size([27793, 10, 10])

        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        x = torch.matmul(attn, value.transpose(-1, -2))

        self.attn = attn.detach()

        x = x.reshape(seq_len, -1)

        # Output layer
        return self.output(x)


class PowerSetFusionWithMoEGating(nn.Module):
    def __init__(
        self,
        num_modalities=6,
        combos=["010000", "000100", "000001", "001010", "101000", "100010", "101010"],
        topk=5,
        # reg_weight=[0.9935, 0.0035, 0.9285, 0.0030, 0.0960, 0.0000],
        # hidden_dim=512,
    ):
        super().__init__()
        self.num_combos = len(combos)
        weight = torch.randn(self.num_combos, 1)
        logger.info(f"weight :{weight}")
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.num_modalities = num_modalities
        self.all_combo_masks = self.build_combo_masks(num_modalities)  # [6, 63]
        if combos:
            self.combos = combos
            self.combo_indices = self.select_combo_indices(combos)
        else:
            self.combo_indices = list(range(2**num_modalities - 1))
        logger.info(f"combo_indices :{self.combo_indices}")
        self.topk = topk

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
        """
        inputs: list of 6 tensors [B, D]
        Returns:
            fused: [B, 6*D]
            reg_loss: MSE between modal_weights and target_modal_weights
        """
        assert len(inputs) == self.num_modalities
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

        self.gph_emb = GAT(
            n_units=self.n_units,
            n_heads=self.n_heads,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            instance_normalization=self.args.instance_normalization,
            diag=True,
        )
        self.joint = PowerSetFusionWithMoEGating()
        self.img_fc = MoEAdaptorLayer(layers=[4096, args.feat_dim])
        self.att_txt_fc = MoEAdaptorLayer(layers=[768, args.feat_dim])
        self.rel_txt_fc = MoEAdaptorLayer(layers=[768, args.feat_dim])
        self.att_fc = MoEAdaptorLayer(layers=[1000, args.feat_dim])
        self.rel_fc = MoEAdaptorLayer(layers=[1000, args.feat_dim])
        self.cross_graph_model = GAT(
            n_units=self.n_units,
            n_heads=self.n_heads,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            instance_normalization=self.args.instance_normalization,
            diag=True,
        )

    def forward(
        self,
        e_device,
        input_idx,
        adj,
        img_features=None,
        rel_features=None,
        att_features=None,
        att_text_features=None,
        rel_text_features=None,
    ):
        gph_emb = self.gph_emb(self.entity_emb(input_idx).to(e_device), adj, e_device)
        img_emb, _, _ = self.img_fc(img_features)
        rel_emb, _, _ = self.rel_fc(rel_features)
        rel_text_emb, _, _ = self.rel_txt_fc(rel_text_features)
        att_emb, _, _ = self.att_fc(att_features)
        att_text_emb, _, _ = self.att_txt_fc(att_text_features)
        joint_emb = self.joint(
            [gph_emb, img_emb, rel_emb, rel_text_emb, att_emb, att_text_emb]
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
