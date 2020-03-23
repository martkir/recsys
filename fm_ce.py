__author__ = 'mbarutchiyska'

import torch
import torch.nn as nn
from torch import optim
import math
import mf


class FMCE(nn.Module):

    def __init__(self, x_dim, u_dim, num_factors):
        super(FMCE, self).__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.num_factors = num_factors
        self.linear = nn.Linear(self.x_dim + self.u_dim, 1, bias=True)
        self.V = nn.Parameter(torch.Tensor(self.x_dim + self.u_dim, self.num_factors))
        k = self.x_dim + self.u_dim  # alternative: k = self.num_factors.
        bound = 1 /  math.sqrt(k)
        nn.init.uniform_(self.V, -bound, bound)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, u):
        z = torch.cat([x, u], dim=1)  # (b, z_dim).
        scores_lin = self.linear(z)  # (b, 1).

        V = self.V.unsqueeze(0)  # (1, z_dim, k).
        V = V.repeat(x.shape[0], 1, 1)  # (b, z_dim, k).
        z = z.unsqueeze(1)  # (b, 1, z_dim).

        scores_inter = torch.bmm(z, V)  # (b, 1, k).
        scores_inter = torch.bmm(scores_inter, torch.transpose(V, 1, 2))  # (b, 1, k) * (b, k, z_dim) = (b, 1, z_dim).
        scores_inter = torch.bmm(scores_inter, torch.transpose(z, 1, 2))  # (b, 1, z_dim) * (b, z_dim, 1) = (b, 1, 1).
        scores_inter = scores_inter.reshape(scores_inter.shape[0], 1)  # (b, 1).

        scores = scores_lin + scores_inter  # (b, 1).
        scores = self.sigmoid(scores)  # (b, 1).

        return scores


class UserItemSampler(mf.UserItemSampler):
    def __init__(self, rels, user_ids, item_ids, device):
        super(UserItemSampler, self).__init__(rels, user_ids, item_ids, device)

    def batch(self, batch_size=1):

        while True:
            u = []
            x = []
            targets = []

            for _ in range(batch_size):
                try:
                    user_id, item_id, rel = next(self.rels)
                    rel = 1 if rel > 0 else 0  # ensure cross entropy is possible.
                    user_vec = self.user_matrix[self.uid2idx[user_id]].reshape(1, self.user_matrix.shape[1])
                    item_vec = self.item_matrix[self.xid2idx[item_id]].reshape(1, self.item_matrix.shape[1])
                    u.append(user_vec)
                    x.append(item_vec)
                    targets.append(torch.tensor(rel, dtype=torch.long).reshape(1, 1))
                except StopIteration:
                    break

            if len(targets) > 0:
                u = torch.cat(u, dim=0)
                x = torch.cat(x, dim=0)
                targets = torch.cat(targets, dim=0).to(device=self.device)
                batch = (x, u, targets.reshape(targets.shape[0], 1))
                yield batch
            else:
                break


class TrainEvalJob(mf.TrainEvalJob):
    def __init__(self, job_id, num_factors, lr, batch_size, num_epochs, use_gpu=True, override=False):
        super(TrainEvalJob, self).__init__(job_id, num_factors, lr, batch_size, num_epochs, use_gpu, override)

        self.model = FMCE(
            x_dim=len(self.train_partition.item_ids),
            u_dim=len(self.train_partition.user_ids),
            num_factors=self.num_factors
        )

        self.model.to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.mse_loss = nn.BCELoss()

    def compute_acc(self, scores, targets):
        preds = scores.flatten().round()
        targets = targets.float().flatten()
        num_correct = (preds == targets).sum().float()
        num_total = torch.tensor(targets.shape[0]).float()
        acc = torch.div(num_correct, num_total)
        return acc

    def reset_sampler(self, name='train'):
        if name == 'train':
            self.train_sampler = UserItemSampler(
                rels=self.train_partition.rels,
                user_ids=self.train_partition.user_ids,
                item_ids=self.train_partition.item_ids,
                device=self.device
            )
        elif name == 'valid':
            self.valid_sampler = UserItemSampler(
                rels=self.valid_partition.rels,
                user_ids=self.train_partition.user_ids,
                item_ids=self.train_partition.item_ids,
                device=self.device
            )
        else:
            raise ValueError('name {} is not valid.'.format(name))

    def train_iter(self, x, u, targets):
        self.model.train()
        scores = self.model.forward(x, u)  # (b, 1).
        loss = self.mse_loss(scores, targets.float())   # targets must be (b, 1) (done by sampler)
        stats = dict()
        stats['train_loss'] = loss.data.cpu().numpy()
        stats['train_acc'] = self.compute_acc(scores, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return stats

    def valid_iter(self, x, u, targets):
        self.model.eval()
        scores = self.model.forward(x, u)
        loss = self.mse_loss(scores, targets.float())
        stats = dict()
        stats['valid_loss'] = loss.data.cpu().numpy()
        stats['valid_acc'] = self.compute_acc(scores, targets)
        return stats


def test():

    job = TrainEvalJob(
        job_id=1,
        num_factors=4,
        lr=0.01,
        batch_size=256,
        num_epochs=5,
        use_gpu=False,
        override=False
    )

    job.run()

test()