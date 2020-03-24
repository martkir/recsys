import torch
import torch.nn as nn
import math
import mf


class FM(nn.Module):

    def __init__(self, x_dim, u_dim, num_factors):
        super(FM, self).__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.num_factors = num_factors
        self.linear = nn.Linear(self.x_dim + self.u_dim, 1, bias=True)
        self.V = nn.Parameter(torch.Tensor(self.x_dim + self.u_dim, self.num_factors))
        k = self.x_dim + self.u_dim  # alternative: k = self.num_factors.
        bound = 1 /  math.sqrt(k)
        nn.init.uniform_(self.V, -bound, bound)

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

        scores = scores_lin + scores_inter

        return scores


class TrainEvalJob(mf.TrainEvalJob):
    def __init__(self, index_path, num_factors, lr, batch_size, num_epochs, use_gpu=True, override=False):
        super(TrainEvalJob, self).__init__(index_path, num_factors, lr, batch_size, num_epochs, use_gpu, override)

        self.model = FM(
            x_dim=len(self.train_partition.item_ids),
            u_dim=len(self.train_partition.user_ids),
            num_factors=self.num_factors
        )
        self.model.to(device=self.device)