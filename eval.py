
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import click
from tabulate import tabulate

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
# todo: change the whole thing with job_ids - everything should be done assuming job_configs.
# todo: more clear error messages.
# todo: nlargest should be nsmallest depending on metric.
# todo: add histogram of performances.


def get_job_str(job_config):
    return ' '.join(['{}:{}'.format(k, v) for k, v in job_config.items()])


def get_job(job_config):
    model = job_config['model']
    job_kwargs = {k: v for k, v in job_config.items() if k != 'model'}
    if model == 'mf':
        import mf
        return mf.TrainEvalJob(**job_kwargs)
    elif model == 'fm':
        import fm
        return fm.TrainEvalJob(**job_kwargs)


class EvalHist(object):
    def __init__(self, schema_path, metric, index_path, bins):
        self.schema_path = schema_path
        self.metric = metric
        self.index_path = index_path
        self.bins = bins
        self.table = EvalTable(
            schema_path=self.schema_path,
            agg_option='all',
            metric=self.metric,
            index_path=self.index_path
        )
        self.table.create()
        self.vals = np.array(self.table.table[self.metric])

    def save(self, hist_path):
        plt.hist(self.vals, bins=self.bins)
        plt.savefig(hist_path)
        print('Finished creating histogram. Saved at {}'.format(hist_path))


class EvalTable(object):
    def __init__(self, schema_path, agg_option, metric, index_path):
        self.schema_path = schema_path
        self.agg_option = agg_option
        self.metric = metric
        self.index_path = index_path
        self.schema = json.load(open(self.schema_path, 'r'))
        try:
            self.index = json.load(open(self.index_path, 'r'))
        except FileNotFoundError:
            self.index = {}
        self.check()
        self.table = None

    def check(self):
        missing_jobs = []
        for model in self.schema.keys():
            for job_config in self.schema[model]['job_configs']:
                job_config_str = ' '.join(['{}:{}'.format(k, v) for k, v in job_config.items()])
                if job_config_str not in self.index:
                    missing_jobs.append(job_config)

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
        for model in self.schema.keys():
            for job_config in self.schema[model]['job_configs']:
                job_id = self.index[get_job_str(job_config)]
                results = pd.read_csv('jobs/{}/results.csv'.format(job_id))
                best_result = results.nsmallest(1, self.metric).reset_index(drop=True)  # ensure index is 0.
                row = pd.DataFrame({hparam: [v] for hparam, v in job_config.items()})
                row = pd.concat([row, best_result], axis=1)
                table = pd.concat([table, row], axis=0, sort=False)
        return table

    def best(self):
        table = pd.DataFrame({})
        for model in self.schema.keys():
            model_table = pd.DataFrame({})
            for job_config in self.schema[model]['job_configs']:
                job_id = self.index[get_job_str(job_config)]
                results = pd.read_csv('jobs/{}/results.csv'.format(job_id))
                best_result = results.nsmallest(1, self.metric).reset_index(drop=True)  # ensure index is 0.
                row = pd.DataFrame({hparam: [v] for hparam, v in job_config.items()})
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
        with open(table_path, 'w+') as file:
            table_str = tabulate(self.table, tablefmt="pipe", headers="keys", showindex=False)
            file.write(table_str)
        print('Finished creating table. Saved at {}'.format(table_path))

# python eval.py --compare all --metric valid_mse --schema schemas/test.json --save figs/table_all.md
# python eval.py --compare best --metric valid_mse --schema schemas/test.json --save figs/table_best.md
# python eval.py --hist 20 --metric valid_mse --schema schemas/hist_test.json --save figs/hist_test.png

@click.command()
@click.option('--compare', type=str)  # options: all, best.
@click.option('--hist', type=int)
@click.option('--metric', type=str, default='valid_mse')
@click.option('--schema', type=str)
@click.option('--index', type=str, default='jobs/index.json')
@click.option('--save', type=str)
def main(compare, hist, metric, schema, index, save):
    if compare:
        table = EvalTable(
            schema_path=schema,
            agg_option=compare,
            metric=metric,
            index_path=index
        )
        table.create()
        table.save(table_path=save)

    if hist:
        plot = EvalHist(
            schema_path=schema,
            metric=metric,
            index_path=index,
            bins=hist
        )
        plot.save(hist_path=save)


if __name__ == '__main__':
    main()