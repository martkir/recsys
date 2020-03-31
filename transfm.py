import torch.nn as nn
import torch
import math


"""
comp(u, x1) = dist(u + u', x1) -> but dist(x
comp(u, x1), u * x2, x1 * x2
"""


class TransFM(nn.Module):

    def __init__(self, x_dim, u_dim, num_factors):
        super(TransFM, self).__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.num_factors = num_factors
        self.linear = nn.Linear(2 * self.x_dim + self.u_dim, 1, bias=True)
        self.V = nn.Parameter(torch.Tensor(2 * self.x_dim + self.u_dim, self.num_factors))  # category-value embeddings.
        self.T = nn.Parameter(torch.Tensor(2 * self.x_dim + self.u_dim, self.num_factors))  # category-trans embeddings.

    def init_params(self):
        # todo: write about initialization.
        k = 2 * self.x_dim + self.u_dim  # alternative: k = self.num_factors.
        bound = 1 / math.sqrt(k)
        nn.init.uniform_(self.V, -bound, bound)  # todo: how is this done in place?
        nn.init.uniform_(self.T, -bound, bound)  # todo: check if this works as expected.

    def forward(self, u, x_prev, x_pos, x_neg):
        vv = torch.sum(torch.mul(self.V, self.V), dim=1, keepdim=True)  # (z_dim, k) -> (z_dim, 1)
        vv = vv.unsqueeze(0).repeat(u.shape[0], 1, 1)  # -> (b, z_dim, 1)
        tt = torch.sum(torch.mul(self.T, self.T), dim=1, keepdim=True)
        tt = tt.unsqueeze(0).repeat(u.shape[0], 1, 1)  # -> (b, z_dim, 1)
        vt = torch.sum(torch.mul(self.V, self.T), dim=1, keepdim=True)  # (z_dim, k) -> (z_dim, 1).
        vt = vt.unsqueeze(0).repeat(u.shape[0], 1, 1)  # (b, z_dim, 1).

        terms = {}
        diag = {}

        for pol in ['pos', 'neg']:

            if pol == 'pos':
                z = torch.cat([u, x_prev, x_pos], dim=1)  # (b, z_dim).
            else:
                z = torch.cat([u, x_prev, x_neg], dim=1)

            terms[pol] = []
            vx = torch.matmul(z, self.V)  # (b, z_dim) * (z_dim, k) -> (b, k)
            tx = torch.matmul(z, self.T)  # (b, z_dim) * (z_dim, k) -> (b, k).
            z_sum = torch.sum(z, dim=1, keepdim=True)  # (b, z_dim) -> (b, 1).

            lin_term = self.linear(z)  # (b, z_dim) -> (b, 1)

            term_1_a = z_sum
            term_1_b = torch.bmm(z.unsqueeze(1), vv).squeeze(2)  # (b, 1, z_dim) * (b, z_dim, 1) -> (b, 1)
            term_1 = torch.mul(term_1_a, term_1_b)  # (b, 1).
            terms[pol].append(term_1)

            term_2_a = z_sum
            term_2_b = torch.bmm(z.unsqueeze(1), tt).squeeze(2)
            term_2 = torch.mul(term_2_a, term_2_b)  # (b, 1)
            terms[pol].append(term_2)

            term_3 = term_2
            terms[pol].append(term_3)

            term_4_a = z_sum # (b, 1).
            term_4_b = torch.bmm(z.unsqueeze(1), vt).squeeze(2)  # (b, 1, z_dim) * (b, z_dim, 1) -> (b, 1).
            term_4 = 2 * torch.mul(term_4_a, term_4_b)
            terms[pol].append(term_4)

            term_5 = 2 * torch.sum(torch.mul(vx, vx), dim=1, keepdim=True)  # -> (b, 1).
            terms[pol].append(term_5)

            print('5: ', term_5.shape)

            # (b, 1, k) * (b, k, 1) -> (b, 1)
            term_6 = 2 * torch.bmm(tx.unsqueeze(1), vx.unsqueeze(2)).squeeze(2)
            terms[pol].append(term_6)

            diag[pol] = torch.sum(torch.mul(tx, tx), dim=1, keepdim=True)  # -> (b, k) -> (b, 1)

        preds = {}
        for pol in ['pos', 'neg']:
            preds[pol] = lin_term + 0.5 * (terms[pol][0] + terms[pol][1] + terms[pol][2] + terms[pol][3] - terms[pol][4] -
                                terms[pol][5]) - 0.5 * diag[pol]

        return preds['pos'], preds['neg']


def test():
    model = TransFM(
        x_dim=10,
        u_dim=10,
        num_factors=3
    )

    batch_size = 30
    u_dim = 10
    x_dim = 10

    u = torch.randn(batch_size, u_dim)
    x_prev = torch.randn(batch_size, x_dim)
    x_pos = torch.randn(batch_size, x_dim)
    x_neg = torch.randn(batch_size, x_dim)
    pos_preds, neg_preds = model.forward(u, x_prev, x_pos, x_neg)
    print(pos_preds.shape)

test()