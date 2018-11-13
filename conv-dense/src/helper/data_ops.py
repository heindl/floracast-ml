"""
Handle data and perform operations on it. Includes sorting, reformatting, etc.
"""
import collections
from collections import defaultdict
import numpy as np

def shuffle_batches(batches):
    # TODO: this should be an inplace sort
    ### First shuffle the order of the batches
    num_batches = len(batches[0])
    indices = np.random.permutation(np.arange(int(num_batches)))
    Xb = batches[0]
    yb = batches[1]
    ### Then shuffle within batches
    for i in range(num_batches):
        cur_length = len(Xb[i])
        indices = np.random.permutation(np.arange(cur_length))
        Xb[i] = Xb[i][indices]
        yb[i] = yb[i][indices]

    return Xb, yb

def batch_data(unbatched_data, batch_size):
    features = unbatched_data[0]
    labels = unbatched_data[1]
    d_features = defaultdict(list)
    d_labels = defaultdict(list)

    for x,l in zip(features,labels):
        k = len(x)
        if not d_features[k] or len(d_features[k][-1])>=batch_size:
            d_features[k].append([])
            d_labels[k].append([])
        d_features[k][-1].append(x)
        d_labels[k][-1].append(l)

    ###Squish into a list
    batched_features = []
    batched_labels = []
    for (k_feat,v_feat),(k_label,v_label) in zip(d_features.iteritems(), d_labels.iteritems()):
        for b_feat,b_label in zip(v_feat,v_label):
             batched_features.append(b_feat)
             batched_labels.append(b_label)

    batched_features = np.array(map(np.array, batched_features))
    batched_labels = np.array(map(np.array, batched_labels))

    return batched_features, batched_labels

def normalize_zmuv(data):
    X,y = data

    X_centered = X - np.mean(X, axis=0)
    X_normal = X_centered/np.std(X, axis=0)

    return X_normal, y

def normalize_min_max(data):
    X,y = data

    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    X_normal = (X - x_min) / (x_max-x_min)

    return X_normal, y


def split_data(data, test_split=0.2):
    X,y = data

    num_batches = len(X)
    indices = np.random.permutation(np.arange(int(num_batches)))
    X = X[indices]
    y = y[indices]

    n = len(X)
    training_size = int(n*(1-test_split))
    X_tr = X[:training_size]
    X_te = X[training_size:]
    Y_tr = y[:training_size]
    Y_te = y[training_size:]

    return (X_tr,Y_tr),(X_te,Y_te)
