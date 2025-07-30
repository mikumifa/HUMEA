import argparse
from datetime import datetime
import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
from loguru import logger
import torch.nn.functional as F
from layers import MultiLossLayer
from loss import ial_loss, icl_loss
from model import MultiModalEncoder, list_rebul_sort
from utils import (
    csls_sim,
    get_adjr,
    get_ids,
    load_att_text,
    load_attr,
    load_img,
    load_rel_text,
    load_relation,
    pairwise_distances,
    read_raw_data,
)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"{timestamp}.log"
logger.add("log/" + log_file)


def load_img_features(ent_num, file_dir):
    # load images features
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        img_vec_path = (
            "data/mmkb-datasets/"
            + filename
            + "/"
            + filename
            + "_id_img_feature_dict.pkl"
        )
    else:
        img_vec_path = None

    img_features = load_img(ent_num, img_vec_path)
    return img_features


def load_att_txt_features(ent_num, file_dir):
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        att_text_vec_path = (
            "data/mmkb-datasets/" + filename + "/" + "attribute_feature_dict.pkl"
        )  # roberta-base
        att_txt_features = load_att_text(ent_num, att_text_vec_path)
    else:
        att_txt_features = None
    return att_txt_features


def load_rel_txt_features(ent_num, file_dir):
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        rel_text_vec_path = (
            "data/mmkb-datasets/" + filename + "/" + "triples_feature_dict.pkl"
        )
    else:
        pass

    rel_txt_features = load_rel_text(ent_num, rel_text_vec_path)
    return rel_txt_features


class HUMEA:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = self.parse_options(self.parser)
        self.set_seed(self.args.seed, True)
        self.device = torch.device("cuda")
        self.init_data()
        self.init_model()

    @staticmethod
    def parse_options(parser):
        parser.add_argument(
            "--file_dir",
            type=str,
            default="data/DBP15K/zh_en",
            required=False,
        )
        parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
        parser.add_argument("--seed", type=int, default=2021, help="random seed")
        parser.add_argument(
            "--epochs", type=int, default=1000, help="number of epochs to train"
        )
        parser.add_argument("--check_point", type=int, default=100, help="check point")
        parser.add_argument(
            "--hidden_units",
            type=str,
            default="300,300,300",
        )
        parser.add_argument(
            "--heads",
            type=str,
            default="2,2",
        )
        parser.add_argument(
            "--instance_normalization",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--lr", type=float, default=0.005, help="initial learning rate"
        )

        parser.add_argument(
            "--dropout", type=float, default=0.0, help="dropout rate for layers"
        )
        parser.add_argument(
            "--attn_dropout",
            type=float,
            default=0.0,
            help="dropout rate for gat layers",
        )
        parser.add_argument(
            "--gph_dim",
            type=int,
            default=300,
        )
        parser.add_argument(
            "--n_exp",
            type=int,
            default=5,
        )
        parser.add_argument(
            "--csls", action="store_true", default=False, help="use CSLS for inference"
        )
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        parser.add_argument(
            "--il_start", type=int, default=500, help="If Il, when to start?"
        )
        parser.add_argument("--bsize", type=int, default=7500, help="batch size")
        parser.add_argument(
            "--tau",
            type=float,
            default=0.1,
            help="the temperature factor of contrastive loss",
        )
        parser.add_argument(
            "--tau2",
            type=float,
            default=1,
            help="the temperature factor of alignment loss",
        )
        parser.add_argument(
            "--feat_dim", type=int, default=300, help="the hidden size of img feature"
        )
        parser.add_argument(
            "--use_project_head",
            action="store_true",
            default=False,
            help="use projection head",
        )

        parser.add_argument(
            "--zoom", type=float, default=0.1, help="narrow the range of losses"
        )
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]")
        parser.add_argument("--train_ill_path", type=str, default="", help="")

        return parser.parse_args()

    @staticmethod
    def set_seed(seed, cuda=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def init_data(self):
        # Load data
        lang_list = [1, 2]
        file_dir = self.args.file_dir
        device = self.device
        self.ent2id_dict, self.ills, self.triples = read_raw_data(file_dir, lang_list)
        left_ents = get_ids(os.path.join(file_dir, "ent_ids_1"))
        right_ents = get_ids(os.path.join(file_dir, "ent_ids_2"))
        self.ENT_NUM = len(self.ent2id_dict)

        np.random.shuffle(self.ills)
        self.img_features = F.normalize(
            torch.Tensor(load_img_features(self.ENT_NUM, file_dir)).to(device)
        )
        self.att_txt_features = F.normalize(
            torch.Tensor(load_att_txt_features(self.ENT_NUM, file_dir)).to(device)
        )
        self.rel_txt_features = F.normalize(
            torch.Tensor(load_rel_txt_features(self.ENT_NUM, file_dir)).to(device)
        )
        # train/val/test split
        if self.args.train_ill_path:
            logger.info(f"use {self.args.train_ill_path}")
            self.train_ill = np.load(self.args.train_ill_path)
        else:
            self.train_ill = np.array(
                self.ills[: int(len(self.ills) // 1 * self.args.rate)], dtype=np.int32
            )

        self.test_ill = np.array(
            self.ills[int(len(self.ills) // 1 * self.args.rate) :], dtype=np.int32
        )

        self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).to(device)
        self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).to(device)
        self.left_non_train = list(set(left_ents) - set(self.train_ill[:, 0].tolist()))
        self.right_non_train = list(
            set(right_ents) - set(self.train_ill[:, 1].tolist())
        )
        self.rel_features = torch.Tensor(
            load_relation(self.ENT_NUM, self.triples, 1000)
        ).to(device)

        self.att_features = torch.Tensor(
            load_attr(
                [
                    os.path.join(file_dir, "training_attrs_1"),
                    os.path.join(file_dir, "training_attrs_2"),
                ],
                self.ENT_NUM,
                self.ent2id_dict,
                1000,
            )
        ).to(device)
        self.adj = get_adjr(self.ENT_NUM, self.triples, norm=True).to(self.device)

    def init_model(self):
        self.multimodal_encoder = MultiModalEncoder(
            args=self.args,
            ent_num=self.ENT_NUM,
        ).to(self.device)
        self.multi_loss_layer = MultiLossLayer(loss_num=6).to(self.device)
        self.align_multi_loss_layer = MultiLossLayer(loss_num=6).to(self.device)

        self.params = [
            {
                "params": list(self.multimodal_encoder.parameters())
                + list(self.multi_loss_layer.parameters())
                + list(self.align_multi_loss_layer.parameters())
            }
        ]
        self.optimizer = optim.AdamW(self.params, lr=self.args.lr)
        self.criterion_cl = icl_loss(
            device=self.device,
            tau=self.args.tau,
            ab_weight=0.5,
            n_view=2,
        )
        self.criterion_align = ial_loss(
            device=self.device,
            tau=self.args.tau2,
            ab_weight=0.5,
            zoom=self.args.zoom,
            reduction=self.args.reduction,
        )

    def semi_supervised_learning(self):
        with torch.no_grad():
            (
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                att_text_emb,
                rel_text_emb,
                joint_emb,
            ) = self.multimodal_encoder(
                self.device,
                self.input_idx,
                self.adj,
                self.img_features,
                self.rel_features,
                self.att_features,
                att_text_features=self.att_txt_features,
                rel_text_features=self.rel_txt_features,
            )

            final_emb = F.normalize(joint_emb)

        distance_list = []
        for i in np.arange(0, len(self.left_non_train), 1000):
            d = pairwise_distances(
                final_emb[self.left_non_train[i : i + 1000]],
                final_emb[self.right_non_train],
            )
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        return preds_l, preds_r

    def inner_view_loss(
        self, gph_emb, rel_emb, att_emb, att_text_emb, rel_text_emb, img_emb, train_ill
    ):
        loss_GCN = self.criterion_cl(gph_emb, train_ill)
        loss_rel = self.criterion_cl(rel_emb, train_ill)
        loss_att = self.criterion_cl(att_emb, train_ill)
        loss_img = self.criterion_cl(img_emb, train_ill)
        loss_att_text = self.criterion_cl(att_text_emb, train_ill)
        loss_rel_text = self.criterion_cl(rel_text_emb, train_ill)

        total_loss = self.multi_loss_layer(
            [loss_GCN, loss_rel, loss_att, loss_att_text, loss_rel_text, loss_img]
        )
        return total_loss

    def kl_alignment_loss(
        self,
        joint_emb,
        gph_emb,
        rel_emb,
        att_emb,
        att_text_emb,
        rel_text_emb,
        img_emb,
        train_ill,
    ):
        zoom = self.args.zoom
        loss_GCN = self.criterion_align(gph_emb, joint_emb, train_ill)
        loss_rel = self.criterion_align(rel_emb, joint_emb, train_ill)
        loss_att = self.criterion_align(att_emb, joint_emb, train_ill)
        loss_img = self.criterion_align(img_emb, joint_emb, train_ill)
        loss_att_text = self.criterion_align(att_text_emb, joint_emb, train_ill)
        loss_rel_text = self.criterion_align(rel_text_emb, joint_emb, train_ill)

        total_loss = (
            self.align_multi_loss_layer(
                [loss_GCN, loss_rel, loss_att, loss_att_text, loss_rel_text, loss_img]
            )
            * zoom
        )
        return total_loss

    def train(self):
        logger.info(f"{self.args}")
        t_total = time.time()
        bsize = self.args.bsize
        device = self.device

        self.input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(device)
        for epoch in range(0, self.args.epochs):
            self.multimodal_encoder.train()
            self.multi_loss_layer.train()
            self.align_multi_loss_layer.train()
            self.optimizer.zero_grad()

            (
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                att_text_emb,
                rel_text_emb,
                joint_emb,
            ) = self.multimodal_encoder(
                self.device,
                self.input_idx,
                self.adj,
                self.img_features,
                self.rel_features,
                self.att_features,
                att_text_features=self.att_txt_features,
                rel_text_features=self.rel_txt_features,
            )
            loss_all = []
            np.random.shuffle(self.train_ill)
            if epoch <= self.args.il_start:
                if epoch % 50 == 0:
                    k_o = 5
                    k_v = k_o - epoch // 50
                    if k_v < 1:
                        self.train_list = self.train_ill
                    else:
                        self.train_list = list_rebul_sort(
                            self.train_ill, joint_emb, k=k_v
                        )
                        print("K value is :" + str(k_v))
            else:
                self.train_list = self.train_ill

            for si in np.arange(0, self.train_list.shape[0], bsize):
                in_loss = self.inner_view_loss(
                    gph_emb,
                    rel_emb,
                    att_emb,
                    att_text_emb,
                    rel_text_emb,
                    img_emb,
                    self.train_list[si : si + bsize],
                )
                align_loss = self.kl_alignment_loss(
                    joint_emb,
                    gph_emb,
                    rel_emb,
                    att_emb,
                    att_text_emb,
                    rel_text_emb,
                    img_emb,
                    self.train_list[si : si + bsize],
                )
                loss_all.append(in_loss + align_loss)

            del gph_emb, rel_emb, att_emb, att_text_emb, rel_text_emb, img_emb
            torch.cuda.empty_cache()

            for si in np.arange(0, self.train_list.shape[0], bsize):
                loss_joi = self.criterion_cl(
                    joint_emb, self.train_list[si : si + bsize]
                )
                loss_all.append(loss_joi)

            sum(loss_all).backward()
            self.optimizer.step()
            if epoch % self.args.check_point == 0:
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                self.test(epoch)

        print("[optimization finished!]")
        print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))

    def test(self, epoch):
        with torch.no_grad():
            self.multimodal_encoder.eval()
            self.multi_loss_layer.eval()
            self.align_multi_loss_layer.eval()

            (
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                att_text_emb,
                rel_text_emb,
                joint_emb,
            ) = self.multimodal_encoder(
                self.device,
                self.input_idx,
                self.adj,
                self.img_features,
                self.rel_features,
                self.att_features,
                att_text_features=self.att_txt_features,
                rel_text_features=self.rel_txt_features,
            )
            logger.info(f"{self.multimodal_encoder.joint.latest_weights}")
            top_k = [1, 5, 10]
            results = {
                "gph_emb": gph_emb,
                "img_emb": img_emb,
                "att_emb": att_emb,
                "joint_emb": joint_emb,
                "rel_emb": rel_emb,
                "att_text_emb": att_text_emb,
                "rel_text_emb": rel_text_emb,
            }
            for name, emb in results.items():
                acc_l2r, acc_r2l = self.evaluate_embedding(
                    emb, f"epoch {epoch} - {name}", top_k
                )

            del (
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                att_text_emb,
                rel_text_emb,
                joint_emb,
            )

    def evaluate_embedding(self, emb, name, top_k):
        emb = F.normalize(emb)
        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        mean_l2r = mean_r2l = mrr_l2r = mrr_r2l = 0.0

        distance = pairwise_distances(emb[self.test_left], emb[self.test_right])
        distance = 1 - csls_sim(1 - distance, self.args.csls_k)

        for idx in range(self.test_left.shape[0]):
            _, indices = torch.sort(distance[idx, :], descending=False)
            rank = (indices == idx).nonzero().squeeze().item()
            mean_l2r += rank + 1
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1

        for idx in range(self.test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=False)
            rank = (indices == idx).nonzero().squeeze().item()
            mean_r2l += rank + 1
            mrr_r2l += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_r2l[i] += 1

        mean_l2r /= self.test_left.size(0)
        mean_r2l /= self.test_right.size(0)
        mrr_l2r /= self.test_left.size(0)
        mrr_r2l /= self.test_right.size(0)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / self.test_left.size(0), 4)
            acc_r2l[i] = round(acc_r2l[i] / self.test_right.size(0), 4)

        logger.info(
            f"{name} avg: acc@{top_k}={((acc_l2r + acc_r2l) / 2)}, mr={(mean_l2r + mean_r2l) / 2:.3f}, mrr={(mrr_l2r + mrr_r2l) / 2:.3f}"
        )

        return (acc_l2r, acc_r2l)


if __name__ == "__main__":
    model = HUMEA()
    model.train()
