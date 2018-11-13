"""
Can compute accuracy, loss, etc. for a model.
"""

import tensorflow as tf

class EvalModel(object):
    def __init__(self):
        self.loss = None
        self.correct = None
        self.accuracy = None

    def build(self, model, y_t, num_classes):
        """
        Model must have been built before this is called.
        """
        if model.output is None:
            raise ValueError('model.output not intialized before building EvalModel.')

        one_hot = tf.one_hot(y_t, num_classes)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.output['soft'], labels=one_hot))
        self.correct = tf.equal(model.output['pred'], tf.argmax(one_hot, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))