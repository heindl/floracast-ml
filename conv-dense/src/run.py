"""
Main script for running experiments.
"""

import click
import yaml
import os
import datetime

import tune
import load

# TODO: remove after done testing one dataset
from helper import loaders


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('params_path', type=click.Path(exists=True))
@click.option('--gpu', default=0, type=click.INT)
def main(data_dir, params_path, gpu):
    if gpu < 0 or gpu > 7:
        raise ValueError('GPU value must be in range [0,7], has value "%s"' % gpu)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

    print 'Starting: %s' % datetime.datetime.now()
    
    with open(params_path, 'r') as fin:
        params = yaml.safe_load(fin)

    data_dict = load.Loader(correct_dims=('?','?','$',1)).load(data_dir, params['data_sets'])

    del params['data_sets']

    tuner = tune.HyperTuner(params, data_dict).tune()

    print 'Ending: %s' % datetime.datetime.now()


if __name__=='__main__':
    main()
