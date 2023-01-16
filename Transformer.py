import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAtt (nn.Module):
    def __init__(self, kqv_dim, w_dim):
        super(MultiHeadAtt, self).__init__()
        self.W_K = nn.Linear(kqv_dim, w_dim)
        self.W_Q = nn.Linear(kqv_dim, w_dim)
        self.K_norm = nn.LayerNorm(kqv_dim)
        self.Q_norm = nn.LayerNorm(kqv_dim)

    def forward(self, k, q, v):
        k = k.permute(0, 2, 1)
        q = q.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        # k.shape = (B, 256, 1)
        # q.shape = (B, 256, 1)
        k = k.squeeze(2)
        q = q.squeeze(2)

        w_k = self.W_K(k)
        w_k = w_k.unsqueeze(1)
        k = k.unsqueeze(2)
        k_bar = k.matmul(w_k)
        k_bar = self.K_norm(k_bar)

        w_q = self.W_Q(q)
        w_q = w_q.unsqueeze(1)
        q = q.unsqueeze(2)

        q_bar = q.matmul(w_q)
        q_bar = self.K_norm(q_bar)

        kq = torch.softmax(k_bar.matmul(q_bar),dim=1)
        out = kq.matmul(v)

        return out


class MultiLinearPerceptron(nn.Module):
    def __init__(self, dim, depth):
        super(MultiLinearPerceptron, self).__init__()
        self.FC = nn.Linear(dim, dim)

    def forward(self, x):
        y = self.FC(x)
        return y


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth):
        super(TransformerEncoder, self).__init__()
        self.MHA = MultiHeadAtt(dim, dim)
        self.Norm = nn.LayerNorm(dim)
        self.MLP = MultiLinearPerceptron(dim, depth)

    def forward(self, r, g, b, h_r, h_g, h_b, k_r, k_g, k_b):
        mha_r = self.MHA(h_r, k_r, r)
        r = r.permute(0, 2, 1)
        r = mha_r + r
        r = r.squeeze(2)
        r = self.Norm(r)
        r = self.MLP(r) + r

        mha_g = self.MHA(h_g, k_g, g)
        g = g.permute(0, 2, 1)
        g = mha_g + g
        g = g.squeeze(2)
        g = self.Norm(g)
        g = self.MLP(g) + g

        mha_b = self.MHA(h_b, k_b, b)
        b = b.permute(0, 2, 1)
        b = mha_b + b
        b = b.squeeze(2)
        b = self.Norm(b)
        b = self.MLP(b) + b

        # y = torch.cat((r, g, b), dim=1)

        return r, g, b

