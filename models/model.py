import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.ComplexLayers import CompEncoderBlock
from layers.Embed import PatchEmbedding, DataEmbedding
from layers.GCN import homo_hete_gnn
from layers.SelfAttention import AttentionBlock



def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class Block(nn.Module):
    def __init__(self, configs):
        super(Block, self).__init__()
        self.configs = configs
        self.k = configs.top_k
        self.seq_len = configs.seq_len

        # self.graph_block = nn.ModuleList()
        # for i in range(self.k):
        #     self.graph_block.append(homo_hete_gnn(configs))
        # self.graph_block = homo_hete_gnn(configs)

        self.attention_block = AttentionBlock(
            configs.d_model,
            configs.d_ff,
            configs.n_heads,
            configs.dropout,
        )

        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # graph_block
            # x = self.graph_block[i](x)

            # padding
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).reshape(-1, period, N)

            # attention
            out = self.norm(self.attention_block(out))
            out = self.gelu(out)

            # reshape back
            out = out.reshape(B, -1, period, N).reshape(B, -1, N)
            res.append(out[:, : self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class T_Block(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.model = nn.ModuleList([Block(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)
        self.graph_block = homo_hete_gnn(configs)

    def forward(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = self.enc_embedding(x_enc)
        enc_out = self.graph_block(x_enc)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out


class F_Block(nn.Module):
    def __init__(self, configs):
        super(F_Block, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.model = CompEncoderBlock(configs)

    def forward(self, x_enc):
        freq = torch.fft.rfft(
            x_enc - torch.mean(x_enc, dim=1, keepdim=True), dim=1
        )  # [B, L, n_vars], dtype=torch.complex
        freq = freq / x_enc.shape[1]

        # Frequency Normalization
        means = torch.mean(freq, dim=1)
        freq_abs = torch.abs(freq)
        stdev = torch.sqrt(torch.var(freq_abs, dim=1, keepdim=True) + 1e-5)
        freq = (freq - means.unsqueeze(1).detach()) / stdev

        freq_pred = self.model(freq)

        # Frequency De-Normalization
        freq_pred = freq_pred * stdev
        freq_pred = freq_pred + means.unsqueeze(1).detach()

        freq_pred = freq_pred * x_enc.shape[1]
        pred_seq = torch.fft.irfft(freq_pred, dim=1)
        pred_seq = pred_seq + torch.mean(x_enc, dim=1, keepdim=True)
        return pred_seq


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        self.t_model = T_Block(configs)
        self.f_model = F_Block(configs)

    def forward(self, x_enc):
        t_out = self.t_model(x_enc)
        f_out = self.f_model(x_enc)

        out = (t_out + f_out) / 2
        return out
