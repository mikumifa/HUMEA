#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class MultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """

    def __init__(
        self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False
    ):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head, 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)
        if init is not None and diag:
            init(self.w)
            stdv = 1.0 / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, adj, mh_device):
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            edge = adj._indices()
            if self.diag:
                h = torch.mul(input, self.w[i])
            else:
                h = torch.mm(input, self.w[i])

            edge_h = torch.cat(
                (h[edge[0, :], :], h[edge[1, :], :]), dim=1
            )  # edge: 2*D x E

            torch.backends.cudnn.enabled = False  # new add

            edge_e = torch.exp(
                -self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze())
            )  # edge_e: 1 x E

            e_rowsum = self.special_spmm(
                edge,
                edge_e,
                torch.Size([N, N]),
                torch.ones(size=(N, 1)).to(mh_device)
                if next(self.parameters()).is_cuda
                else torch.ones(size=(N, 1)),
            )  # e_rowsum: N x 1
            edge_e = F.dropout(
                edge_e, self.attn_dropout, training=self.training
            )  # edge_e: 1 x E

            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)

            output.append(h_prime.unsqueeze(0))

        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias  # type: ignore
        else:
            return output

    def __repr__(self):
        if self.diag:
            return (
                self.__class__.__name__
                + " ("
                + str(self.f_out)
                + " -> "
                + str(self.f_out)
                + ") * "
                + str(self.n_head)
                + " heads"
            )
        else:
            return (
                self.__class__.__name__
                + " ("
                + str(self.f_in)
                + " -> "
                + str(self.f_out)
                + ") * "
                + str(self.n_head)
                + " heads"
            )


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)  # spmm does sparse matrix multiplication
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer"""

    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor"""

    def __init__(self, n_exps=5, layers=[], dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList(
            [PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)]
        )
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, r=None, train=None, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits).to(x.device) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits
        if r is not None:
            gates = F.softmax(logits / torch.sigmoid(r), dim=-1)
        else:
            gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x, r=None):
        gates = self.noisy_top_k_gating(x, r, self.training)  # (B, n_E)
        expert_outputs = [
            self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)
        ]  # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return (
            multiple_outputs.sum(dim=-2),
            expert_outputs,
            gates,
        )  # multiple_outputs是混合的，expert_outputs是多个，gates是权重


class MultiLossLayer(nn.Module):
    """
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """

    def __init__(self, loss_num, device=None):
        super(MultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(
            torch.zeros(
                self.loss_num,
            ),
            requires_grad=True,
        )

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss
