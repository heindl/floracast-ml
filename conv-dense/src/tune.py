"""
Takes a model, a trainer and a set of hyperparameters to tune.
"""

import tensorflow as tf
import numpy as np
import sys

import train
import test
from models.jiffy import JiffyModel
from models.eval import EvalModel

from helper import data_ops
from helper import data_wrapper as dw
from helper.nearest_neighbor import NearestNeighborClassifier
from helper.hyperparams import HyperParameterWrapper

class HyperTuner(object):
    def __init__(self, params, data_dict):
        self.hyperparams = HyperParameterWrapper(params)
        self.data_dict = data_dict

    def tune(self):
        self.sess = tf.Session()


        # Create train/test objects
        tr = train.ModelTrainer(self.sess)
        te = test.ModelTester(self.sess)


        for dp in self.hyperparams.data_params:
            print '***'
            print '+:', dp
            list_data_wrapper = self.prep_data(dp)
            #self.run_baselines(list_data_wrapper)

            for hp in self.hyperparams:
                for dw in list_data_wrapper:
                    sys.stdout.flush()
                    print
                    print '========'
                    print '#:', hp
                    print '$:', dw.name
                    print '========'
                    model_tensors = self.build_model_tensors(dw)
                    
                    # Build model and eval
                    model_eval = EvalModel()
                    model = JiffyModel(model_eval, dw.width, dw.num_classes, hp['num_filters'], hp['embed_size'], hp['pool_size'], hp['dropout_rate'])

                    model.build(model_tensors['X_t'], model_tensors)
                    model_eval.build(model, model_tensors['y_t'], dw.num_classes)

                    # Train and test model
                    tr.train(model, model_tensors, dw, hp['num_epochs'], hp['init_learning_rate'], hp['dropout_rate'], hp['pool_size'])
                    te.test(model, model_tensors, dw, hp['pool_size'])

        self.sess.close()

    def prep_data(self, data_params):
        list_data_wrapper = []

        for k,v in self.data_dict.iteritems():
            list_data_wrapper.append(dw.TimeseriesDataWrapper(k, v, normalization=data_params['data_normalization'], batch_size=data_params['data_batch_size'], split_ratio=data_params['data_split_ratio']))

        return list_data_wrapper

    def build_model_tensors(self, data_wrapper):
        model_tensors = {}

        model_tensors['X_t'] = tf.placeholder('float', [None, None, data_wrapper.width, 1])
        model_tensors['y_t'] = tf.placeholder('uint8', [None])

        model_tensors['pool_k_size_t'] = tf.placeholder(tf.int32, shape = 4)
        model_tensors['pool_strides_t'] = tf.placeholder(tf.int32, shape = 4)
        model_tensors['pool_pad_t'] = tf.placeholder(tf.int32, shape = [4, 2])
        model_tensors['apply_dropout'] = tf.placeholder(tf.bool)

        return model_tensors

    def compute_baseline_one_nearest_neighbor(self, dw):
        data = dw.get_data()
        print dw.name, data.keys()
        (X_tr,y_tr), (X_te,y_te) = data['train'],data['test']

        # Flatten features and channels into a single dim
        X_tr_flat = []
        for x in X_tr:
            x = x.flatten()
            X_tr_flat.append(x)

        X_tr_flat = np.array(X_tr_flat)

        X_te_flat = []
        for x in X_te:
            x = x.flatten()
            X_te_flat.append(x)

        X_te_flat = np.array(X_te_flat)

        acc_nn_te = NearestNeighborClassifier().compute_one_nearest_neighbor_accuracy_pad(X_tr_flat, y_tr, X_te_flat, y_te)

        return acc_nn_te

    def run_baselines(self, list_data_wrapper):
        for dw in list_data_wrapper:
            # compute initial 1nn for comparison
            acc_nn_te = self.compute_baseline_one_nearest_neighbor(dw)
            print '^:', dw.name
            print '&: 1NN Accuracy (input features): Test=%f' % acc_nn_te


