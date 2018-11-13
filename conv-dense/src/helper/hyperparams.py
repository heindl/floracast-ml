"""
Handling input hyperparameter specifications and exposing all combinations
as an iterable object.
"""

import numpy as np
import itertools

class HyperParameterWrapper(object):
    def __init__(self, hyperparams):
        self.data_params, hyperparams_ext = self.extract_data_params(hyperparams)
        hyperparams_prep = self.prep_param(hyperparams_ext)
        hyperparams_prod = itertools.product(*hyperparams_prep)

        self.hyperparams_all = map(dict, hyperparams_prod)

    def __iter__(self):
        return hyperparam_generator(self.hyperparams_all)

    def is_iterable(self, v):
        if isinstance(v, str): return False
        try:
            _ = (x for x in v)
            return True
        except TypeError:
            return False

    def eval_param(self, v):
        try:
            v = eval(v)
        except Exception: 
            pass

        return v

    def extract_data_params(self, hyperparams):
        data_params = {}
        hyperparams_ext = {}

        # Extract data params
        for k, v in hyperparams.iteritems():
            if 'data_' in k:
                data_params[k] = v
            else:
                hyperparams_ext[k] = v

        # Create product of all combinations
        data_params_prep = self.prep_param(data_params)
        data_params_prod = itertools.product(*data_params_prep)

        data_params = map(dict, data_params_prod)

        return data_params, hyperparams_ext

    def prep_param(self, hyperparams):
        hyperparams_prep = []
        for k,v in hyperparams.iteritems():
            v = self.eval_param(v)
            if self.is_iterable(v):
                v = map(lambda x: (k,x), v) 
            else:
                v = [(k,v)]

            hyperparams_prep.append(v)

        return hyperparams_prep


def hyperparam_generator(hyperparams_all):
    for hp in hyperparams_all:
        yield hp
