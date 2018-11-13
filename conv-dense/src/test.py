"""
Used to evaluate a trained model.
"""

import tensorflow as tf
import numpy as np

from helper.nearest_neighbor import NearestNeighborClassifier
from helper import tf_help

class ModelTester(object):
    def __init__(self, sess):
        self.sess = sess


    def test(self, model, model_tensors, data_wrapper, pool_size):
        data = data_wrapper.get_data() 
        Xb_tr, Xb_te = data['train_batch'],data['test_batch']

        # compute cnn accuracy and embeddings for train and test sets
        with self.sess.as_default():
            acc_cnn_tr, embed_tr, y_tr = tf_help.compute_eval_step(self.sess, model, model_tensors, Xb_tr, pool_size)
            acc_cnn_te, embed_te, y_te = tf_help.compute_eval_step(self.sess, model, model_tensors, Xb_te, pool_size)

        acc_nn_te = NearestNeighborClassifier().compute_one_nearest_neighbor_accuracy(embed_tr, y_tr, embed_te, y_te)

        print ''
        print '@: CNN Accuracy: Train=%f, Test=%f | 1NN Accuracy: Test=%f' % (acc_cnn_tr, acc_cnn_te, acc_nn_te)
