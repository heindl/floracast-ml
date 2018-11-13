"""
Takes a model, data and a set of hyperparameters and returns a trained model.
"""

import tensorflow as tf
import time as time
import sys

from helper import data_ops
from helper import tf_help
from helper.nearest_neighbor import NearestNeighborClassifier

REG_UPDATE_RATE = 200
LONG_UPDATE_RATE = 1000


class ModelTrainer(object):
    def __init__(self, sess):
        self.sess = sess

    def train(self, model, model_tensors, data_wrapper, num_epochs, init_learning_rate, dropout_rate, pool_size):
        # TODO: Make hyperparams a passed in dict (shorten function prototype)
        # TODO: Support choosing between Adam and SGD w/ Nest. Momentum
        optimizer = self.build_optimizer(model, init_learning_rate)

        self.sess.run(tf.global_variables_initializer())
        with self.sess.as_default():
            data = data_wrapper.get_data()
            Xb_tr = data['train_batch']
            Xb_te = data['test_batch']

            start = time.time()
            for epoch in range(num_epochs+1):
                acc_epoch, loss_epoch, embed_tr_epoch, y_tr_epoch = tf_help.compute_train_step(self.sess, model, model_tensors, Xb_tr, optimizer, pool_size)

                if epoch % LONG_UPDATE_RATE == 0:
                    acc_cnn_te, embed_te, y_te = tf_help.compute_eval_step(self.sess, model, model_tensors, Xb_te, pool_size)
                    acc_nn_te = NearestNeighborClassifier().compute_one_nearest_neighbor_accuracy(embed_tr_epoch, y_tr_epoch, embed_te, y_te)
                    print 'Epoch: %5d -- CNN Accuracy: Train=%f, Loss=%f, Test=%f | 1NN Accuracy: Test=%f' % (epoch, acc_epoch, loss_epoch, acc_cnn_te, acc_nn_te)

                elif epoch % REG_UPDATE_RATE == 0:
                    print 'Epoch: %5d -- CNN Accuracy: Train=%f, Loss=%f | ' % (epoch, acc_epoch, loss_epoch),
                    print 'Mean Time Per Epoch=%f' % ((time.time()-start)/float(REG_UPDATE_RATE))
                    start = time.time()
                    sys.stdout.flush()


                data_wrapper.shuffle(['train_batch'])

    def build_optimizer(self, model, init_learning_rate):
        return tf.train.AdamOptimizer(init_learning_rate).minimize(model.eval.loss)

    
