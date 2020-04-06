import os
import json
import shutil
import pandas as pd
import datasets


class Jobs(object):
    def __init__(self):
        self.index_path = 'jobs/index.json'
        if os.path.isfile(self.index_path):  # update index if a job directory was deleted by user:
            self.index = json.load(open(self.index_path))
            job_dir_names = next(os.walk(os.path.dirname(self.index_path)))[1]
            valid_index_ids = set([int(job_dir_name) for job_dir_name in job_dir_names])
            for k in list(self.index.keys()):
                if self.index[k] not in valid_index_ids:
                    del self.index[k]
        else:
            os.makedirs(self.index_path)
            self.index = {}
        json.dump(self.index, open(self.index_path, 'w+'))

    def add(self, job_id_str):
        if job_id_str not in self.index:
            job_id = len(self.index)
            self.index[job_id_str] = job_id
            job_dir = os.path.join('jobs', '{}'.format(job_id))
            checkpoints_dir = os.path.join(job_dir, 'checkpoints')
            os.makedirs(checkpoints_dir)
            json.dump(self.index, open(self.index_path, 'w+'), indent=4)
        else:
            raise ValueError('Job {} already exists. Rerun job by setting override to TRUE'.format(job_id_str))

    def remove(self, job_id_str):
        if job_id_str in self.index:
            job_id = self.index[job_id_str]
            job_dir = os.path.join('jobs', '{}'.format(job_id))
            shutil.rmtree(job_dir)
            del self.index[job_id_str]
            json.dump(self.index, open(self.index_path, 'w+'), indent=4)

    def is_job(self, job_id_str):
        if job_id_str in self.index:
            return True
        else:
            return False

    def get_results_path(self, job_id_str):
        if job_id_str in self.index:
            job_id = self.index[job_id_str]
            job_dir = os.path.join('jobs/{}'.format(job_id))
            results_path = os.path.join(job_dir, 'results.csv')
            return results_path

    def get_checkpoints_dir(self, job_id_str):
        if job_id_str in self.index:
            job_id = self.index[job_id_str]
            job_dir = os.path.join('jobs/{}'.format(job_id))
            checkpoints_dir = os.path.join(job_dir, 'checkpoints')
            return checkpoints_dir


class Data(object):
    def __init__(self, name):
        self.name = name
        self.root_dir = 'data/rs_data'
        if not os.path.isdir(self.root_dir):
            os.makedirs(self.root_dir)
        try:
            present = set(next(os.walk('data/rs_data'))[1])
        except StopIteration:
            present = set()
        if self.name not in present:
            datasets.create(self.name, self.root_dir)

    def get_ratings(self):
        path = 'data/rs_data/{}/ratings.csv'.format(self.name)  # todo: change to ratings.csv
        return pd.read_csv(path, sep=',')
