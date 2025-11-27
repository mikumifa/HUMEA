import pickle
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def read_raw_data(file_dir, l=[1, 2], reverse=False):
    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups

    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids

    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in l])
    ills = read_file([file_dir + "/ill_ent_ids"])
    triples = read_file([file_dir + "/triples_" + str(i) for i in l])
    rel_size = max([t[1] for t in triples]) + 1
    reverse_triples = []
    r_hs, r_ts = {}, {}  # type:ignore
    for h, r, t in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
        if reverse:
            reverse_r = r + rel_size
            reverse_triples.append((t, reverse_r, h))
            if reverse_r not in r_hs:
                r_hs[reverse_r] = set()
            if reverse_r not in r_ts:
                r_ts[reverse_r] = set()
            r_hs[reverse_r].add(t)
            r_ts[reverse_r].add(h)
    if reverse:
        triples = triples + reverse_triples
    assert len(r_hs) == len(r_ts)
    return ent2id_dict, ills, triples


def get_adjr(ent_size, triples, norm=False):
    """
    已经包含关系逆操作
    """
    print("getting a sparse tensor r_adj...")
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 0
        M[(tri[0], tri[2])] += 1
    ind, val = [], []
    for fir, sec in M:
        ind.append((fir, sec))
        ind.append((sec, fir))  # 关系逆
        val.append(M[(fir, sec)])
        val.append(M[(fir, sec)])
    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)
    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix(
            (val, (ind[:, 0], ind[:, 1])),  # type: ignore
            shape=(ent_size, ent_size),
            dtype=np.float32,
        )
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    else:
        M = torch.sparse_coo_tensor(
            torch.LongTensor(ind).t(),
            torch.FloatTensor(val),
            torch.Size([ent_size, ent_size]),
        )
        return M


def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.
    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.
    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = torch.mean(torch.topk(sim_mat, k)[0], 1)
    nearest_values2 = torch.mean(torch.topk(sim_mat.t(), k)[0], 1)
    csls_sim_mat = 2 * sim_mat.t() - nearest_values1
    csls_sim_mat = csls_sim_mat.t() - nearest_values2
    return csls_sim_mat


def loadfile(fn, num=1):
    print("loading a file..." + fn)
    ret = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split("\t")
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ids(fn):
    ids = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split("\t")
            ids.append(int(th[0]))
    return ids


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                ent2id[th[1]] = int(th[0])
    return ent2id


def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    for i in range(min(topA, len(fre))):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr


def load_relation(e, KG, topR=1000):
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(KG)[:, 1]  # 中间的关系构成列表
    top_rels = Counter(rels).most_common(topR)
    # return a list of the first n elements in descending order of occurrence
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.0
            rel_mat[o][rel_index_dict[r]] += 1.0
    return np.array(rel_mat)


def load_att_text(e_num, path):
    att_text_dict = pickle.load(open(path, "rb"))
    att_text_np = np.array(list(att_text_dict.values()))
    mean = np.mean(att_text_np, axis=0)
    std = np.std(att_text_np, axis=0)
    att_text_emb = np.array(
        [
            att_text_dict[i]
            if i in att_text_dict
            else np.random.normal(mean, std, mean.shape[0])
            for i in range(e_num)
        ]
    )
    print("%.2f%% entities have attribute text" % (100 * len(att_text_dict) / e_num))
    return att_text_emb


def load_rel_text(e_num, path):
    rel_text_dict = pickle.load(open(path, "rb"))
    rel_text_np = np.array(list(rel_text_dict.values()))
    mean = np.mean(rel_text_np, axis=0)
    std = np.std(rel_text_np, axis=0)
    rel_text_emb = np.array(
        [
            rel_text_dict[i]
            if i in rel_text_dict
            else np.random.normal(mean, std, mean.shape[0])
            for i in range(e_num)
        ]
    )
    print("%.2f%% entities have attribute text" % (100 * len(rel_text_dict) / e_num))
    return rel_text_emb


def load_img(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    img_embd = np.array(
        [
            img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0])
            for i in range(e_num)
        ]
    )
    print("%.2f%% entities have images" % (100 * len(img_dict) / e_num))
    return img_embd
