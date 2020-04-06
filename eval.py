
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import click
from tabulate import tabulate
import db
import os
from mf import get_job_id_str

"""
tasks:
- change a training evaluation job to update the "index" file.
- modify TrainEvalJob to create index if it doesn't exist.
- modify TrainEvalJob to create populate index automatically.

design remarks:
- the missing index file is correct. job with given index must have been run first in order for it to exist.
"""

"""
todos:
- Exceptions have to be a bit more granular.
- histogram hast be be styled.
- add comments.
"""

# todo: removing a directory should only be possible from the command line to ensure the index gets updated.
# todo: add job id to summary table.
# todo: figure out why summary df rows are identical - shouldn't be the case.
# todo: nlargest should be nsmallest depending on metric.
# todo: add histogram of performances.


def get_job(job_config):
    if job_config['model_name'] == 'mf':
        import mf
        return mf.TrainEvalJob(**job_config)
    if job_config['model_name'] == 'fm':
        import fm
        return fm.TrainEvalJob(**job_config)
    elif job_config['model_name'] == 'transfm':
        import transfm
        return transfm.TrainEvalJob(**job_config)


class EvalHist(object):
    def __init__(self, config_path, metric, bins):
        self.config_path = config_path
        self.metric = metric
        self.bins = bins
        self.table = EvalTable(
            config_path=self.config_path,
            agg_option='all',
            metric=self.metric,
        )
        self.table.create()
        self.vals = np.array(self.table.table[self.metric])

    def save(self, hist_path):
        plt.hist(self.vals, bins=self.bins)
        plt.savefig(hist_path)
        print('Finished creating histogram. Saved at {}'.format(hist_path))


class EvalTable(object):
    def __init__(self, config_path, agg_option, metric):
        self.config_path = config_path
        self.agg_option = agg_option
        self.metric = metric
        self.config = json.load(open(self.config_path, 'r'))
        self.jobs_db = db.Jobs()
        self.check()
        self.table = None

    def check(self):
        missing_jobs = []
        for config in self.config:
            job_id_str = get_job_id_str(**config)
            if not self.jobs_db.is_job(job_id_str):
                missing_jobs.append(config)

        if len(missing_jobs) > 0:
            print('Table cannot be created. The results of the following jobs are missing: ')
            print(json.dumps(missing_jobs, indent=4))
            while True:
                ans = input('Run all of the above jobs? [y/n] ')
                if ans not in ['y', 'n']:
                    print('Input {} is not a valid response.'.format(ans))
                else:
                    break
            if ans == 'y':
                for job_config in missing_jobs:
                    job = get_job(job_config)
                    job.run()
                print('Finished running all missing jobs. Re-running script should work now.')
            exit()

    def all(self):
        table = pd.DataFrame({})
        for config in self.config:
            job_id_str = get_job_id_str(**config)
            results_path = self.jobs_db.get_results_path(job_id_str)
            results = pd.read_csv(results_path)
            best_result = results.nsmallest(1, self.metric).reset_index(drop=True)  # ensure index is 0.
            row = pd.DataFrame({hparam: [v] for hparam, v in config.items()})
            row = pd.concat([row, best_result], axis=1)
            table = pd.concat([table, row], axis=0, sort=False)
        return table

    def best(self):
        table = pd.DataFrame({})
        model_table = pd.DataFrame({})
        for config in self.config:
            job_id_str = get_job_id_str(**config)
            results_path = self.jobs_db.get_results_path(job_id_str)
            results = pd.read_csv(results_path)
            best_result = results.nsmallest(1, self.metric).reset_index(drop=True)  # ensure index is 0.
            row = pd.DataFrame({hparam: [v] for hparam, v in config.items()})
            row = pd.concat([row, best_result], axis=1)
            model_table = pd.concat([model_table, row], axis=0, sort=False)
        table = pd.concat([table, model_table.nsmallest(1, self.metric)], axis=0, sort=False)
        return table

    def create(self):
        if self.agg_option == 'all':
            self.table = self.all()
        else:
            self.table = self.best()

    def save(self, table_path):
        if not os.path.isdir(os.path.dirname(table_path)):
            os.makedirs(table_path)
        with open(table_path, 'w+') as file:
            table_str = tabulate(self.table, tablefmt="pipe", headers="keys", showindex=False)
            file.write(table_str)
        print('Finished creating table. Saved at {}'.format(table_path))

# python eval.py --compare all --config results/transfm/config.json --save results/transfm/all.md
# python eval.py --compare all --config results/transfm/config.json --save results/transfm/best.md

@click.command()
@click.option('--compare', type=str)  # options: all, best.
@click.option('--metric', type=str, default='valid_loss')
@click.option('--hist', type=int)
@click.option('--config', type=str)
@click.option('--save', type=str)
def main(compare, metric, hist, config, save):
    if compare:
        table = EvalTable(
            config_path=config,
            agg_option=compare,
            metric=metric,
        )
        table.create()
        table.save(table_path=save)

    if hist:
        plot = EvalHist(
            config_path=config,
            metric=metric,
            bins=hist
        )
        plot.save(hist_path=save)


if __name__ == '__main__':
    main()