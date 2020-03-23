__author__ = 'mbarutchiyska'

import wget
from zipfile import ZipFile
import os
import pandas as pd


def get_data(dataset_name):
    if dataset_name == 'ml-100k':
        return MovieLens100k(dir_path='data')
    elif dataset_name == 'ml-1m':
        return MovieLens1M(dir_path='data')
    else:
        raise ValueError('Dataset name {} is not a recognized dataset.'.format(dataset_name))


class MovieLens100k(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.raw_dir_path = os.path.join(self.dir_path, 'raw_data/ml-100k')
        self.rs_dir_path = os.path.join(self.dir_path, 'rs_data/ml-100k')
        self.raw_data_path = os.path.join(self.raw_dir_path, 'ml-100k.zip')
        self.rels = []

        if not os.path.isdir(self.raw_dir_path):
            os.makedirs(self.raw_dir_path)
            wget.download(
                url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                out=self.raw_data_path
            )

        if not os.path.isdir(self.rs_dir_path):
            os.makedirs(self.rs_dir_path)
            self.create()

        rels_df = pd.read_csv(os.path.join(self.rs_dir_path, 'rels.csv'))
        rels_dict = rels_df.to_dict(orient='list')

        for user_id, item_id, rel in zip(rels_dict['user_id'], rels_dict['item_id'], rels_dict['rel']):
            self.rels.append((user_id, item_id, rel))

    def create(self):
        rels_dict = {'user_id': [], 'item_id': [], 'rel': []}

        with ZipFile(self.raw_data_path, mode='r') as archive:
            archive.printdir()
            for line in archive.open('ml-100k/u.data'):
                record = line.decode('utf-8').split('\t')

                user_id = int(record[0])
                item_id = int(record[1])
                rel = int(record[2])
                rels_dict['user_id'].append(user_id)
                rels_dict['item_id'].append(item_id)
                rels_dict['rel'].append(rel)

        df = pd.DataFrame(rels_dict)
        df.to_csv(os.path.join(self.rs_dir_path, 'rels.csv'), index=False, mode='w')


class MovieLens1M(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.raw_dir_path = os.path.join(self.dir_path, 'raw_data/ml-1m')
        self.rs_dir_path = os.path.join(self.dir_path, 'rs_data/ml-1m')
        self.raw_data_path = os.path.join(self.raw_dir_path, 'ml-1m.zip')
        self.rels = []

        if not os.path.isdir(self.raw_dir_path):
            os.makedirs(self.raw_dir_path)
            wget.download(
                url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                out=self.raw_data_path
            )

        if not os.path.isdir(self.rs_dir_path):
            os.makedirs(self.rs_dir_path)
            self.create()

        rels_df = pd.read_csv(os.path.join(self.rs_dir_path, 'rels.csv'))
        rels_dict = rels_df.to_dict(orient='list')

        for user_id, item_id, rel in zip(rels_dict['user_id'], rels_dict['item_id'], rels_dict['rel']):
            self.rels.append((user_id, item_id, rel))

    def create(self):
        rels_dict = {'user_id': [], 'item_id': [], 'rel': []}

        with ZipFile(self.raw_data_path, mode='r') as archive:
            archive.printdir()

            for line in archive.open('ml-1m/ratings.dat'):
                record = line.decode('utf-8').split('::')
                user_id = int(record[0])
                item_id = int(record[1])
                rel = int(record[2])
                rels_dict['user_id'].append(user_id)
                rels_dict['item_id'].append(item_id)
                rels_dict['rel'].append(rel)

        df = pd.DataFrame(rels_dict)
        df.to_csv(os.path.join(self.rs_dir_path, 'rels.csv'), index=False, mode='w')
