import torch.nn as nn
import torch
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from torch import optim
import random
from collections import defaultdict
from collections import OrderedDict
import db
from mf import get_job_id_str
from mf import set_device


def parse(df, partition_method='transfm'):
    user_ids = set(list(df['user_id']))
    if partition_method == 'transfm':
        obs = {}
    else:
        obs = []
    for user_id in user_ids:
        df_user = df[df['user_id'] == user_id]
        df_user = df_user.sort_values('time', ascending=True)  # oldest first.
        item_ids = np.array(df_user['item_id'])
        for i in range(1, len(item_ids)):
            item_id_prev = item_ids[i - 1]
            item_id_pos = item_ids[i]
            if partition_method == 'transfm':
                if user_id not in obs:
                    obs[user_id] = []
                obs[user_id].append((item_id_prev, item_id_pos))
            else:
                obs.append((user_id, item_id_prev, item_id_pos))
    return obs


class SubsetTransFM(object):
    def __init__(self):
        self.user_ids = set()
        self.item_ids = set()
        self.obs = []

    def add(self, user_id, prev_item_id, pos_item_id):
        self.user_ids.add(user_id)
        self.item_ids.add(prev_item_id)
        self.item_ids.add(pos_item_id)
        self.obs.append((user_id, prev_item_id, pos_item_id))


class TransFMPartition(object):
    def __init__(self, obs, partition_method='transfm', split_prob=0.8, shuffle=False, seed=0):
        self.kind = partition_method
        self.split_prob = split_prob
        self.shuffle = shuffle
        self.seed = seed
        self.train = SubsetTransFM()
        self.valid = SubsetTransFM()
        if partition_method == 'transfm':
            train_item_ids = set()
            valid_candidates = set()
            for user_id in obs:
                num_obs = len(obs[user_id])
                for prev_item_id, pos_item_id in obs[user_id][:num_obs - 1]:
                    self.train.add(user_id, prev_item_id, pos_item_id)
                    train_item_ids.add(prev_item_id)
                    train_item_ids.add(pos_item_id)
                valid_cand_prev_item_id = obs[user_id][num_obs - 1][0]
                valid_cand_pos_item_id = obs[user_id][num_obs - 1][1]
                valid_candidates.add((user_id, valid_cand_prev_item_id, valid_cand_pos_item_id))
            for (user_id, prev_item_id, pos_item_id) in valid_candidates:
                if prev_item_id in train_item_ids and pos_item_id in train_item_ids:
                    self.valid.add(user_id, prev_item_id, pos_item_id)
                else:
                    print('removed: ', 'user_id: {} prev_id: {} pos_id: {}'.format(user_id, prev_item_id, pos_item_id))
        else:
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(obs)
            split_point = int(len(obs) * self.split_prob)
            for user_id, prev_item_id, pos_item_id in obs[:split_point]:
                self.train.add(user_id, prev_item_id, pos_item_id)
            for user_id, prev_item_id, pos_item_id in obs[split_point:]:
                if user_id in self.train.user_ids and prev_item_id in self.train.item_ids \
                        and pos_item_id in self.train.item_ids:
                    self.valid.add(user_id, prev_item_id, pos_item_id)
                else:
                    print('removed: ', 'user_id: {} prev_id: {} pos_id: {}'.format(user_id, prev_item_id, pos_item_id))


class ValidSampler(object):
    def __init__(self, valid_obs, user_item_matrix):
        self._size = len(valid_obs)
        self.obs = iter(valid_obs)
        self.user_item_matrix = user_item_matrix

    def __len__(self):
        return self._size

    def batch(self, batch_size=1):

        while True:
            u = []
            x_prev = []
            x_pos = []

            for _ in range(batch_size):
                try:
                    user_id, prev_item_id, pos_item_id = next(self.obs)
                    user_vec = self.user_item_matrix.get_user_sparse(user_id)
                    prev_item_vec = self.user_item_matrix.get_item_sparse(prev_item_id)
                    pos_item_vec = self.user_item_matrix.get_item_sparse(pos_item_id)  # shape (1, num_items).
                    u.append(user_vec)
                    x_prev.append(prev_item_vec)
                    x_pos.append(pos_item_vec)
                except StopIteration:
                    break

            if len(u) > 0:
                item_one_hot = self.user_item_matrix.item_one_hot
                neg_indices = np.random.randint(0, item_one_hot.shape[0], size=len(u))
                x_neg = [item_one_hot[neg_item_idx].reshape(1, item_one_hot.shape[1])
                         for neg_item_idx in neg_indices]   # todo: explain why for valid its okay to sample from all item_ids.
                u = torch.cat(u, dim=0)
                x_prev = torch.cat(x_prev, dim=0)
                x_pos = torch.cat(x_pos, dim=0)
                x_neg = torch.cat(x_neg, dim=0)
                batch = (u, x_prev, x_pos, x_neg)
                yield batch
            else:
                break


class TrainSampler(object):
    def __init__(self, obs, user_item_matrix):
        self._size = len(obs)
        self.obs = iter(obs)
        self.user_item_matrix = user_item_matrix

    def __len__(self):
        return self._size

    def batch(self, batch_size=1):

        while True:
            u = []
            x_prev = []
            x_pos = []

            for _ in range(batch_size):
                try:
                    user_id, prev_item_id, pos_item_id = next(self.obs)
                    user_vec = self.user_item_matrix.get_user_sparse(user_id)
                    prev_item_vec = self.user_item_matrix.get_item_sparse(prev_item_id)
                    pos_item_vec = self.user_item_matrix.get_item_sparse(pos_item_id)  # shape (1, num_items).
                    u.append(user_vec)
                    x_prev.append(prev_item_vec)
                    x_pos.append(pos_item_vec)
                except StopIteration:
                    break

            if len(u) > 0:
                item_one_hot = self.user_item_matrix.item_one_hot
                neg_indices = np.random.randint(0, item_one_hot.shape[0], size=len(u))
                x_neg = [item_one_hot[neg_item_idx].reshape(1, item_one_hot.shape[1])
                         for neg_item_idx in neg_indices]
                u = torch.cat(u, dim=0)
                x_prev = torch.cat(x_prev, dim=0)
                x_pos = torch.cat(x_pos, dim=0)
                x_neg = torch.cat(x_neg, dim=0)
                batch = (u, x_prev, x_pos, x_neg)
                yield batch
            else:
                break


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
        k = 2 * self.x_dim + self.u_dim  # alternative: k = self.num_factors.
        bound = 1 / math.sqrt(k)
        nn.init.uniform_(self.V, -bound, bound)
        nn.init.uniform_(self.T, -bound, bound)

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

            # (b, 1, k) * (b, k, 1) -> (b, 1)
            term_6 = 2 * torch.bmm(tx.unsqueeze(1), vx.unsqueeze(2)).squeeze(2)
            terms[pol].append(term_6)

            diag[pol] = torch.sum(torch.mul(tx, tx), dim=1, keepdim=True)  # -> (b, k) -> (b, 1)

        preds = {}
        for pol in ['pos', 'neg']:
            preds[pol] = lin_term + 0.5 * (terms[pol][0] + terms[pol][1] + terms[pol][2] + terms[pol][3] - terms[pol][4] -
                                terms[pol][5]) - 0.5 * diag[pol]

        return preds['pos'], preds['neg']


class SBPRLoss(nn.Module):
    def __init__(self, lin_reg, emb_reg, trans_reg):
        super(SBPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.lin_reg = lin_reg
        self.emb_reg = emb_reg
        self.trans_reg = trans_reg

    def forward(self, pos_preds, neg_preds, lin_params, emb_params, trans_params):
        lin_params = lin_params.flatten()
        emb_params = emb_params.flatten()
        trans_params = trans_params.flatten()
        params = torch.cat([self.lin_reg * lin_params, self.emb_reg * emb_params, self.trans_reg * trans_params])
        l2_reg = torch.sum(torch.mul(params, params))
        contributions = torch.log(1e-10 + self.sigmoid(pos_preds - neg_preds))  # note: 1e-10 for numerical stability.
        loss = -1 * torch.sum(contributions, dim=0, keepdim=False) + l2_reg  # (1, ).
        return loss


class UserItemMatrix(object):
    def __init__(self, user_ids, item_ids, device):
        self.num_users = len(user_ids)
        self.num_items = len(item_ids)
        self.device = device
        self.uid2idx = {user_id: i for i, user_id in enumerate(user_ids)}
        self.xid2idx = {item_id: i for i, item_id in enumerate(item_ids)}
        self.user_one_hot = torch.eye(self.num_users).to(device=self.device)
        self.item_one_hot = torch.eye(self.num_items).to(device=self.device)

    def get_user_sparse(self, user_id):
        user_idx = self.uid2idx[user_id]
        return self.user_one_hot[user_idx, :].reshape(1, self.num_users)

    def get_item_sparse(self, item_id):
        item_idx = self.xid2idx[item_id]
        return self.item_one_hot[item_idx, :].reshape(1, self.num_items)


class TrainEvalJob(object):
    def __init__(self, dataset_name, num_factors, lr, batch_size, num_epochs, lin_reg, emb_reg, trans_reg, partition_method,
                 use_gpu=True, override=False):

        self.dataset_name = dataset_name
        self.num_factors = num_factors
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lin_reg = lin_reg
        self.emb_reg = emb_reg
        self.trans_reg = trans_reg
        self.partition_method = partition_method
        self.use_gpu = use_gpu
        self.override = override

        self.job_id_str = get_job_id_str('transfm', **self.__dict__.copy())
        self.db_jobs = db.Jobs()
        if self.override:
            self.db_jobs.remove(self.job_id_str)
        self.db_jobs.add(self.job_id_str)
        self.results_path = self.db_jobs.get_results_path(self.job_id_str)
        self.checkpoints_dir = self.db_jobs.get_checkpoints_dir(self.job_id_str)
        self.device = set_device(self.use_gpu)
        data = db.Data(self.dataset_name)
        obs = parse(data.get_ratings(), partition_method=self.partition_method)
        self.partition = TransFMPartition(obs, self.partition_method)
        self.user_item_matrix = \
            UserItemMatrix(self.partition.train.user_ids, self.partition.train.item_ids, self.device)

        self.model = TransFM(
            x_dim=len(self.partition.train.item_ids),
            u_dim=len(self.partition.train.user_ids),
            num_factors=self.num_factors
        )

        self.model.to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.sbpr_loss = SBPRLoss(self.lin_reg, self.emb_reg, self.trans_reg)

        print('finished init job.')

    def reset_sampler(self, name='train'):
        if name == 'train':
            self.train_sampler = TrainSampler(
                obs=self.partition.train.obs,
                user_item_matrix=self.user_item_matrix
            )
        elif name == 'valid':
            self.valid_sampler = ValidSampler(
                valid_obs=self.partition.valid.obs,
                user_item_matrix=self.user_item_matrix
            )
        else:
            raise ValueError('name {} is not valid.'.format(name))

    def train_iter(self, u, x_prev, x_pos, x_neg):
        self.model.train()
        pos_preds, neg_preds = self.model.forward(u, x_prev, x_pos, x_neg)  # (b, 1).
        lin_params = self.model.linear.weight.flatten()
        emb_params = self.model.V.flatten()
        trans_params = self.model.T.flatten()
        loss = self.sbpr_loss(pos_preds, neg_preds, lin_params, emb_params, trans_params)
        stats = dict()
        stats['train_loss'] = loss.data.cpu().numpy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return stats

    def valid_iter(self, u, x_prev, x_pos, x_neg):
        self.model.eval()
        pos_preds, neg_preds = self.model.forward(u, x_prev, x_pos, x_neg)
        lin_params = self.model.linear.weight.flatten()
        emb_params = self.model.V.flatten()
        trans_params = self.model.T.flatten()
        loss = self.sbpr_loss(pos_preds, neg_preds, lin_params, emb_params, trans_params)
        stats = dict()
        stats['valid_loss'] = loss.data.cpu().numpy()
        return stats

    def train_epoch(self, current_epoch):
        batch_train_stats = defaultdict(lambda: [])
        epoch_train_stats = OrderedDict({})
        self.reset_sampler('train')
        with tqdm(total=len(self.train_sampler)) as pbar:
            for i, batch in enumerate(self.train_sampler.batch(self.batch_size)):
                u, x_prev, x_pos, x_neg = batch
                output = self.train_iter(u, x_prev, x_pos, x_neg)
                for k, v in output.items():
                    batch_train_stats[k].append(output[k])
                description = \
                    'epoch: {} '.format(current_epoch) + \
                    ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_train_stats.items()])
                pbar.update(self.batch_size)
                pbar.set_description(description)

        for k, v in batch_train_stats.items():
            epoch_train_stats[k] = np.around(np.mean(v), decimals=4)

        return epoch_train_stats

    def valid_epoch(self, current_epoch):
        batch_valid_stats = defaultdict(lambda: [])
        epoch_valid_stats = OrderedDict({})
        self.reset_sampler('valid')

        with tqdm(total=len(self.valid_sampler)) as pbar:
            for i, batch in enumerate(self.valid_sampler.batch(self.batch_size)):
                u, x_prev, x_pos, x_neg = batch
                output = self.valid_iter(u, x_prev, x_pos, x_neg)
                for k, v in output.items():
                    batch_valid_stats[k].append(output[k])
                description = \
                    'epoch: {} '.format(current_epoch) + \
                    ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_valid_stats.items()])
                pbar.update(self.batch_size)
                pbar.set_description(description)

        for k, v in batch_valid_stats.items():
            epoch_valid_stats[k] = np.around(np.mean(v), decimals=4)

        return epoch_valid_stats

    def save_epoch_stats(self, epoch_stats):
        epoch_stats_df = pd.DataFrame.from_dict(epoch_stats)
        if os.path.exists(self.results_path):
            stats_df = pd.read_csv(self.results_path)
            updated_stats_df = pd.concat([stats_df, epoch_stats_df])
            updated_stats_df.to_csv(self.results_path, index=False, mode='w+')
        else:
            epoch_stats_df.to_csv(self.results_path, index=False, mode='w+')

    def save_checkpoint(self, current_epoch):
        state = dict()
        state['network'] = self.model.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        checkpoint_path = os.path.join(self.checkpoints_dir, 'epoch_{}.ckpt'.format(current_epoch))
        torch.save(state, f=checkpoint_path)

    def run(self):
        for current_epoch in range(self.num_epochs):
            epoch_stats = defaultdict(lambda: [])
            epoch_train_stats = self.train_epoch(current_epoch)
            epoch_valid_stats = self.valid_epoch(current_epoch)

            for k in epoch_train_stats.keys():
                epoch_stats[k] = [epoch_train_stats[k]]
            for k in epoch_valid_stats.keys():
                epoch_stats[k] = [epoch_valid_stats[k]]
            self.save_epoch_stats(epoch_stats)
            self.save_checkpoint(current_epoch)

