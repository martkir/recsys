__author__ = 'mbarutchiyska'

import os
import shutil
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import datasets
import json


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


class Partition(object):
    def __init__(self, rels, dataset_type, split_prob, shuffle, seed=0):
        self.rels = rels
        self.dataset_type = dataset_type
        self.split_prob = split_prob
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.rels)

        split_point = int(len(self.rels) * self.split_prob)
        if self.dataset_type == 'train':
            self.rels = self.rels[:split_point]
        elif self.dataset_type == 'valid':
            self.rels = self.rels[split_point:]
        else:
            raise ValueError('Invalid dataset type {}. Accepted values are train, valid.'.format(dataset_type))

        self.user_ids = set()
        self.item_ids = set()

        for user_id, item_id, _ in self.rels:
            self.user_ids.add(user_id)
            self.item_ids.add(item_id)

    def update(self, user_ids, item_ids):
        self.user_ids = user_ids
        self.item_ids = item_ids

        i = 0
        while True:
            user_id, item_id, _ = self.rels[i]
            if user_id not in self.user_ids:
                del self.rels[i]
                i -= 1
            elif item_id not in self.item_ids:
                del self.rels[i]
                i -= 1
            i += 1
            if i == len(self.rels):
                break


class UserItemSampler(object):
    def __init__(self, rels, user_ids, item_ids, device):
        self.rels = iter(rels)

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
                    user_id, item_id, rel = next(self.rels)
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
    def __init__(self, num_factors, lr, batch_size, num_epochs, use_gpu=True, override=False):
        self.index_path = 'jobs/index.json'
        self.num_factors = num_factors
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_gpu = use_gpu
        self.override = override

        if not os.path.isfile(self.index_path):
            json.dump({}, open(self.index_path, 'w+'))
        self.index = json.load(open(self.index_path))

        self.job_id_str = self.get_job_id_str()  # enables inheritance.
        self.job_id = self.index[self.job_id_str] if self.job_id_str in self.index else len(self.index)
        self.job_dir = os.path.join('jobs', '{}'.format(self.job_id))
        self.checkpoints_dir = os.path.join(self.job_dir, 'checkpoints')
        self.results_path = os.path.join(self.job_dir, 'results.csv')

        if self.job_id_str in self.index:
            if self.override:
                shutil.rmtree(self.job_dir)
            else:
                raise ValueError('Job {} already exists. Rerun job by setting override to TRUE'.format(self.job_id_str))

        self.index[self.job_id_str] = self.job_id
        json.dump(self.index, open(self.index_path, 'w+'), indent=4)
        os.makedirs(self.checkpoints_dir)

        # todo: if experiment doesn't finish - you need to have a status variable that says so.
        # allows unfinished experiments to be overriden.

        self.device = torch.device('cpu')
        if self.use_gpu:
            if not torch.cuda.is_available():
                raise Exception('CUDA capable GPU not available.')
            else:
                self.device = torch.device('cuda:{}'.format(0))

        try:
            print('Job initialized with device {}'.format(torch.cuda.get_device_name(self.device)))
        except (AssertionError, ValueError):
            print('Job initialized with CPU.')

        self.dataset = datasets.get_data('ml-100k')
        self.train_partition = Partition(self.dataset.rels, 'train', 0.8, shuffle=True, seed=0)
        self.valid_partition = Partition(self.dataset.rels, 'valid', 0.8, shuffle=True, seed=0)

        # Ensuring the validation set is a subset of the training set:
        user_ids_diff = self.train_partition.user_ids - self.valid_partition.user_ids
        user_ids_diff = user_ids_diff.union(self.train_partition.user_ids.intersection(self.valid_partition.user_ids))
        item_ids_diff = self.train_partition.item_ids - self.valid_partition.item_ids
        item_ids_diff = item_ids_diff.union(self.train_partition.item_ids.intersection(self.valid_partition.item_ids))
        self.valid_partition.update(user_ids_diff, item_ids_diff)

        self.model = MF(
            x_dim=len(self.train_partition.item_ids),
            u_dim=len(self.train_partition.user_ids),
            num_factors=self.num_factors
        )

        self.model.to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()

    def get_job_id_str(self):
        job_id_str = 'model:{} num_factors:{} lr:{} batch_size:{} num_epochs:{}'.\
            format('mf', self.num_factors, self.lr, self.batch_size, self.num_epochs)
        return job_id_str

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
