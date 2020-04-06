__author__ = 'mbarutchiyska'

import os
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import db


def set_device(use_gpu):
    device = torch.device('cpu')
    if use_gpu:
        if not torch.cuda.is_available():
            raise Exception('CUDA capable GPU not available.')
        else:
            device = torch.device('cuda:{}'.format(0))
    try:
        print('Job initialized with device {}'.format(torch.cuda.get_device_name(device)))
    except (AssertionError, ValueError):
        print('Job initialized with CPU.')
    return device


def get_job_id_str(self, model_name, **kwargs):
    return ' '.join(['model:{}'.format(model_name)] + ['{}:{}'.format(k, v) for k, v in kwargs.items()])


def parse(df_ratings):
    # ratings are df header = user_id,item_id,rating,time
    # output: [(user_id, item_id, rating)]
    obs = []
    records = df_ratings.to_dict(orient='records')
    for record in records:
        obs.append((record['user_id'], record['item_id'], record['rating']))
    return obs


class BatchDotProduct(nn.Module):
    def __init__(self):
        super(BatchDotProduct, self).__init__()

    def forward(self, x, y):
        x = x.unsqueeze(1)  # (b, 1, k).
        y = y.unsqueeze(2)  # (b, k, 1).
        scores = torch.bmm(x, y)  # (b, 1, 1).
        scores = scores.reshape(x.shape[0], 1)
        return scores


class MF(nn.Module):
    def __init__(self, x_dim, u_dim, num_factors):
        super(MF, self).__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.num_factors = num_factors
        self.item_embedder = nn.Linear(x_dim, num_factors)
        self.user_embedder = nn.Linear(u_dim, num_factors)
        self.batch_dot = BatchDotProduct()

    def forward(self, x, u):
        x_embed = self.item_embedder(x)  # (b, k).
        u_embed = self.user_embedder(u)  # (b, k).
        scores_inter = self.batch_dot(u_embed, x_embed)  # (b, 1).
        scores = scores_inter
        return scores


class Subset(object):
    # data needed to sample.
    def __init__(self):
        self.user_ids = set()
        self.item_ids = set()
        self.obs = []

    def add(self, user_id, item_id, rating):
        self.obs.append((user_id, item_id, rating))
        self.user_ids.add(user_id)
        self.item_ids.add(item_id)


class Partition(object):
    # approach used to split data into train/ valid sets.
    def __init__(self, obs, split_prob, shuffle, seed=0):
        self.obs = obs
        self.split_prob = split_prob
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.obs)
        split_point = int(len(self.obs) * self.split_prob)
        self.train = Subset()
        self.valid = Subset()
        for user_id, item_id, rating in self.obs[:split_point]:
            self.train.add(user_id, item_id, rating)
        for user_id, item_id, rating in self.obs[split_point:]:
            if user_id in self.train.user_ids and item_id in self.train.item_ids:
                self.valid.add(user_id, item_id, rating)


class UserItemSampler(object):
    def __init__(self, rels, user_ids, item_ids, device):
        self.obs = iter(rels)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.device = device

        self.uid2idx = {user_id: i for i, user_id in enumerate(self.user_ids)}
        self.xid2idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self._size = len(rels)
        self.user_matrix = torch.eye(self.num_users).to(device=self.device)
        self.item_matrix = torch.eye(self.num_items).to(device=self.device)

    def __len__(self):
        return self._size

    def batch(self, batch_size=1):

        while True:
            u = []
            x = []
            targets = []

            for _ in range(batch_size):
                try:
                    user_id, item_id, rel = next(self.obs)
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
                batch = (x, u, targets.flatten())
                yield batch
            else:
                break


class TrainEvalJob(object):
    def __init__(self, model_name, data_name, num_factors, lr, batch_size, num_epochs, use_gpu=True, override=False):
        self.model_name = model_name  # note: adding model name easier for inheritance.
        self.data_name = data_name
        self.num_factors = num_factors
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_gpu = use_gpu
        self.override = override
        self.job_id_str = get_job_id_str(**self.__dict__.copy())

        self.db_jobs = db.Jobs()
        if self.override:
            self.db_jobs.remove(self.job_id_str)
        self.db_jobs.add(self.job_id_str)
        self.results_path = self.db_jobs.get_results_path(self.job_id_str)
        self.checkpoints_dir = self.db_jobs.get_checkpoints_dir(self.job_id_str)
        self.device = set_device(self.use_gpu)

        data = db.Data(self.data_name)
        obs = parse(data.get_ratings())  # [(user_id, item_id, rating)]
        self.partition = Partition(obs, split_prob=0.8, shuffle=True, seed=0)
        self.model = MF(
            x_dim=len(self.partition.train.item_ids),
            u_dim=len(self.partition.train.user_ids),
            num_factors=self.num_factors
        )

        self.model.to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()

    def reset_sampler(self, name='train'):
        if name == 'train':
            self.train_sampler = UserItemSampler(
                rels=self.partition.train.obs,
                user_ids=self.partition.train.user_ids,
                item_ids=self.partition.train.item_ids,
                device=self.device
            )
        elif name == 'valid':
            self.valid_sampler = UserItemSampler(
                rels=self.partition.valid.obs,
                user_ids=self.partition.train.user_ids,
                item_ids=self.partition.train.item_ids,
                device=self.device
            )
        else:
            raise ValueError('name {} is not valid.'.format(name))

    def train_iter(self, x, u, targets):
        self.model.train()
        scores = self.model.forward(x, u)  # (b, 1).
        loss = self.mse_loss(scores.flatten(), targets.float())
        stats = dict()
        stats['train_loss'] = loss.data.cpu().numpy()
        stats['train_mse'] = loss.data.cpu().numpy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return stats

    def valid_iter(self, x, u, targets):
        self.model.eval()
        scores = self.model.forward(x, u)
        loss = self.mse_loss(scores.flatten(), targets.float())
        stats = dict()
        stats['valid_loss'] = loss.data.cpu().numpy()
        stats['valid_mse'] = loss.data.cpu().numpy(),
        return stats

    def train_epoch(self, current_epoch):
        batch_train_stats = defaultdict(lambda: [])
        epoch_train_stats = OrderedDict({})
        self.reset_sampler('train')

        with tqdm(total=len(self.train_sampler)) as pbar:
            for i, batch in enumerate(self.train_sampler.batch(self.batch_size)):
                x, u, targets = batch
                output = self.train_iter(x, u, targets)
                for k, v in output.items():
                    batch_train_stats[k].append(output[k])
                description = \
                    'epoch: {} '.format(current_epoch) + \
                    ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_train_stats.items()])
                pbar.update(targets.shape[0])
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
                x, u, targets = batch
                output = self.valid_iter(x, u, targets)
                for k, v in output.items():
                    batch_valid_stats[k].append(output[k])
                description = \
                    'epoch: {} '.format(current_epoch) + \
                    ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_valid_stats.items()])
                pbar.update(targets.shape[0])
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
