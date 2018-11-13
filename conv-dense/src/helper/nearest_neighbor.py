"""
Performs nearest neighbor classification.
"""
import numpy as np
from sklearn import neighbors

class NearestNeighborClassifier(object):
    def __init__(self):
        pass
    
    def compute_one_nearest_neighbor_accuracy(self, X_tr, y_tr, X_te, y_te):
        knn = neighbors.KNeighborsClassifier(1)
        knn.fit(X_tr, y_tr)
        pred = knn.predict(X_te)

        correct = np.sum(pred==y_te)
        acc = correct / float(len(y_te))

        return acc

    def compute_one_nearest_neighbor_accuracy_pad(self, X_tr, y_tr, X_te, y_te):
        acc_all = 0

        max_len = 0
        for x in X_tr:
            max_len = x.shape[0] if x.shape[0] > max_len else max_len
        for x in X_te:
            max_len = x.shape[0] if x.shape[0] > max_len else max_len

        X_tr_pad = []
        for x in X_tr:
            pad_len = max_len - len(x)
            if pad_len:
                x = np.pad(x, [(0,pad_len)], 'constant')
            X_tr_pad.append(x)

        X_tr_pad = np.array(X_tr_pad)

        X_te_pad = []
        for x in X_te:
            pad_len = max_len - len(x)
            if pad_len:
                x = np.pad(x, [(0,pad_len)], 'constant')
            X_te_pad.append(x)

        X_te_pad = np.array(X_te_pad)

        X_tr = X_tr_pad
        X_te = X_te_pad

        for i in range(len(X_te)): 
            # find the nearest neighbor in training set for test example i
            nn_i = np.argmin(np.sum((X_tr - X_te[i])**2, axis=1))

            # compare labels of current validation data point with nearest neighbor's label
            acc_all += y_tr[nn_i]==y_te[i]

        acc_all = acc_all / float(len(X_te))

        return acc_all
