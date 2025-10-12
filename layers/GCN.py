import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("btdc,dw->btwc", (x, A))
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order, alpha):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.c_in = (order + 1) * c_in
        self.mlp = nn.Linear(self.c_in, c_out)
        self.dropout = dropout
        self.order = order
        self.act = nn.GELU()
        self.alpha = alpha

    def forward(self, x, a):
        # in: [batch, seq_len, d_model, hidden]
        # out: [batch, seq_len, d_model, hidden]
        out = [x]
        x1 = self.nconv(x, a)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.alpha * x + (1 - self.alpha) * self.nconv(x1, a)
            out.append(x2)
            x1 = x2
        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h  # .unsqueeze(1)


class homo_hete_gnn(nn.Module):
    def __init__(self, configs):
        super(homo_hete_gnn, self).__init__()
        self.init_seq_len = configs.seq_len
        self.channels = configs.d_model
        self.dropout = configs.dropout
        self.device = "cuda:" + str(configs.gpu)
        self.tanh = nn.Tanh()
        self.hidden = configs.hidden
        self.gdepth = configs.gdepth
        self.alpha = configs.alpha
        self.k = configs.k

        self.node_vec1 = nn.Parameter(
            torch.randn(self.channels, configs.nvechidden).to(self.device),
            requires_grad=True,
        ).to(self.device)
        self.node_vec2 = nn.Parameter(
            torch.randn(configs.nvechidden, self.channels).to(self.device),
            requires_grad=True,
        ).to(self.device)
        self.gconv = gcn(self.hidden, self.hidden, self.dropout, self.gdepth, self.alpha)

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.act = nn.Tanh()
        self.gelu = nn.GELU()
        self.start_linear = nn.Linear(1, self.hidden)
        self.Linear = nn.Linear(self.hidden, 1)  # map to initial scale
        #self.linear1 = nn.Linear(configs.d_model, configs.enc_in)
        #self.linear2 = nn.Linear(configs.enc_in, configs.d_model)

    # 根据正负掩码对权重矩阵进行正则化，最后将正负部分相加
    def logits_warper(self, adj, mask_pos, mask_neg, filter_value=-float("Inf")):
        mask_pos_inverse = ~mask_pos
        mask_neg_inverse = ~mask_neg
        # Replace values for mask_pos rows
        processed_pos = mask_pos * F.softmax(adj.masked_fill(mask_pos_inverse, filter_value), dim=-1)
        # Replace values for mask_neg rows
        processed_neg = -1 * mask_neg * F.softmax((1 / (adj + 1)).masked_fill(mask_neg_inverse, filter_value), dim=-1)
        # Combine processed rows for both cases
        processed_adj = processed_pos + processed_neg
        return processed_adj

    def add_cross_var_adj(self, adj):
        # k = 5
        k = self.k
        k = min(k, adj.shape[0])
        # 返回值是和adj形状相同的掩码，1代表是每一行的前k个值
        mask_pos = adj >= torch.topk(adj, k=k)[0][..., -1, None]
        # 返回值是和adj形状相同的掩码，1代表是每一行的后k个值
        mask_neg = adj <= torch.kthvalue(adj, k=k)[0][:, None]
        return mask_pos, mask_neg

    def get_var_adj(self):
        adj = F.softmax(F.relu(torch.einsum("td,dm->tm", self.node_vec1, self.node_vec2)), dim=1)
        mask_pos, mask_neg = self.add_cross_var_adj(adj)
        adj = self.logits_warper(adj, mask_pos, mask_neg)
        return adj

    def expand_channel(self, x):
        # x: [batch, seq_len, d_model]
        # out: [batch, seq_len, d_model, hidden]
        x = x.unsqueeze(-1)
        x = self.start_linear(x)
        return x

    def forward(self, x):
        # x: [Batch, seq_len, d_model]
        x = self.expand_channel(x)  # [batch, seq_len, d_model, hidden]
        gcn_adp = self.get_var_adj()
        x = self.gconv(x, gcn_adp) + x
        # x = torch.cat([x_, x], dim=-1)
        x = self.Linear(x).squeeze(-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # [Batch, seq_len, d_model]
