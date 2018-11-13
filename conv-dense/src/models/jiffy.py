"""
Build default jiffy model.
"""

# from __future__ import print
# from __future__ import division

import tensorflow as tf
import numpy as np
import random as rand

from helper.max_pool import max_pool_v2

class JiffyModel(object):
    def __init__(self, model_eval, width, num_classes, num_filters, embed_size, pool_size, dropout_rate):
        self.eval = model_eval
        self.width = width
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate

        self.embedding = None
        self.output = None

    def build(self, X_t, model_tensors):
        # Build network
        layers = {}

        # pad input to multiple of pool_size
        layers['pad1'] = tf.pad(X_t, model_tensors['pool_pad_t'])

        # reshape so that channel dim is known at compile time
        layers['reshape'] = tf.reshape(layers['pad1'], shape=(-1, tf.shape(layers['pad1'])[1], self.width, 1))

        layers['conv1'] = tf.layers.conv2d(inputs=layers['reshape'], filters=self.num_filters, kernel_size=[5,1], padding='same', activation=tf.nn.relu)
        layers['pool1'] = max_pool_v2(layers['conv1'], ksize=model_tensors['pool_k_size_t'], strides=model_tensors['pool_strides_t'], padding='SAME')

        layers['flatten'] = tf.reshape(layers['pool1'], [-1, self.width*self.pool_size*self.num_filters])

        layers['fc1'] = tf.layers.dense(inputs=layers['flatten'], units=self.embed_size, activation=tf.nn.relu)
        layers['fc1_drop'] = tf.layers.dropout(inputs=layers['fc1'], rate=float(self.dropout_rate), training=model_tensors['apply_dropout'])

        layers['logits'] = tf.layers.dense(inputs=layers['fc1_drop'], units=self.num_classes)

        # Outputs
        layers['soft_out'] = tf.nn.softmax(layers['logits'])
        layers['pred'] = tf.argmax(layers['logits'], axis=1)

        # Update outputs
        self.embedding = layers['fc1']
        self.output = {'soft': layers['soft_out'], 'pred': layers['pred']}