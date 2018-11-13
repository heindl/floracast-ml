"""
Wraps data so that batching, shuffling. etc. can be encapsulated.
"""

import numpy as np

from helper import data_ops

class TimeseriesDataWrapper(object):
    def __init__(self, name, data, split=True, normalization='zmuv', batch_size=20, split_ratio=.2):
        """
        :param batch_size: maximum size of a batch; if None then no batching
        """
        self.name = name

        if len(data)==2:
            X = np.array(data[0])
            y = np.array(data[1], dtype=np.uint8)
            self.data = {'all': (X,y)}

            self.num_classes = len(np.unique(y))
            self.width = np.shape(X[0])[1]

        elif len(data)==4:
            split = False
            self.data = {}

            X_tr = np.array(data[0])
            y_tr = np.array(data[1], dtype=np.uint8)
            self.data['train'] = (X_tr,y_tr)

            X_te = np.array(data[2])
            y_te = np.array(data[3], dtype=np.uint8)
            self.data['test'] = (X_te,y_te)

            self.num_classes = len(np.unique(y_tr))
            self.width = np.shape(X_tr[0])[1]
        else:
            raise ValueError('Number of elements in data is %d (should be 2 or 4)' % len(data))

        # TODO: Currently "leaks" info by fitting normalization on train+test (should only fit on train set)
        #if normalization.lower()!='none': self.__normalize(normalization)
        if normalization: self.__normalize(normalization)
        if split: self.__split(split_ratio)
        if batch_size: self.__batch(batch_size)


    def __split(self, split_ratio):
        # TODO: If memory is an issue, delete the 'all' data after split
        X_tr, X_te = data_ops.split_data(self.data['all'], split_ratio)

        self.data['train'] = X_tr
        self.data['test'] = X_te

    def __normalize(self, normalization):
        if normalization=='zmuv':
            normalize_func = data_ops.normalize_zmuv
        elif normalization=='min_max':
            normalize_func = data_ops.normalize_min_max
        else:
            raise ValueError('Normalization type "%s" is not defined' % normalization)

        for k,v in self.data.iteritems():
            self.data[k] = normalize_func(v)

    def __batch(self, batch_size):
        for k,v in self.data.items():
            self.data[k+'_batch'] = data_ops.batch_data(v, batch_size)

    def shuffle(self, shuffle_list=None):
        if shuffle_list:
            for k in shuffle_list:
                data_ops.shuffle_batches(self.data[k])
        else:
            for k,v in self.data.iteritems():
                data_ops.shuffle_batch(v)

    def get_data(self):
        return self.data
