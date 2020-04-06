__author__ = 'mbarutchiyska'

import wget
from zipfile import ZipFile
import os
import pandas as pd
import json


def create(name, rs_dir_path):
    if name == 'ml-100k':
        d = MovieLens100k(rs_dir_path)
        d.create()
    if name == 'ml-1m':
        d = MovieLens1m(rs_dir_path)
        d.create()


class MovieLens100k(object):
    def __init__(self, rs_dir_path):
        self.name = 'ml-100k'
        self.dir_path = os.path.join(rs_dir_path, 'ml-100k')
        self.raw_data_path = os.path.join('data/raw_data', 'ml-100k.zip')
        self.download_url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

    def create(self):
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        wget.download(
            url=self.download_url,
            out=self.raw_data_path
        )
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)
        self.create_ratings()
        self.create_item_categories()
        print('Finished creating {}.'.format(self.name))

    def create_ratings(self):
        lines = ['user_id,item_id,rating,time\n']
        with ZipFile(self.raw_data_path, mode='r') as archive:
            for line in archive.open('ml-100k/u.data'):
                record = line.decode('utf-8').strip('\n').split('\t')
                line = ','.join(record) + '\n'
                lines.append(line)
        with open(os.path.join(self.dir_path, 'ratings.csv'), 'w+') as file:
            file.writelines(lines)

    def create_item_categories(self):
        lines = ['item_id,item_cat_seq\n']
        with ZipFile(self.raw_data_path, mode='r') as archive:
            for line in archive.open('ml-100k/u.item'):
                record = line.decode('iso-8859-1').strip('\n').split('|')
                item_id = int(record[0])
                genres = [int(record[i]) for i in range(5, len(record))]
                line = '{},"{}"'.format(item_id, json.dumps(genres))
                lines.append(line + '\n')
        with open(os.path.join(self.dir_path, 'item_cat_seq.csv'), 'w+') as file:
            file.writelines(lines)


class MovieLens1m(MovieLens100k):
    def __init__(self, rs_dir_path):
        super(MovieLens1m, self).__init__(rs_dir_path)
        self.name = 'ml-1m'
        self.dir_path = os.path.join(rs_dir_path, '{}'.format(self.name))
        self.raw_data_path = os.path.join('data/raw_data', 'ml-1m.zip')
        self.download_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    def create_ratings(self):
        lines = ['user_id,item_id,rating,time\n']
        with ZipFile(self.raw_data_path, mode='r') as archive:
            for line in archive.open('ml-1m/ratings.dat'):
                record = line.decode('utf-8').strip('\n').split('::')
                line = ','.join(record) + '\n'
                lines.append(line)
        with open(os.path.join(self.dir_path, 'ratings.csv'), 'w+') as file:
            file.writelines(lines)

    def create_item_categories(self):
        pass