import models.jiffy
import numpy as np
import random
from models import*
from helper import*
def generate_data(n, width, max_length):
    feature_seq = []
    labels = []
    for i in range(n):
        length = rand.randint(1, max_length)
        new_series = []
        for j in range(length):
            cur_timestep = []
            for k in range(width):
                cur_timestep.append([rand.randint(0,9)])
                #cur_timestep = [[rand.randint(0,9)], [rand.randint(0,9)], [rand.randint(0,9)] ]
            new_series.append(cur_timestep)
        if new_series[0][0][0] == 3:
            print("*****************************************************")
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        feature_seq.append(new_series)
    return(feature_seq, labels)

def split_hot_encode_data(X, Y, max_size, ratio = 0.20):
    ## Input:
    ## X = list of 2D arrays, each array representing Time Series,
    ## Y = 1D vector of corresponding labels
    ## max_size = length of the longest Time Series
    ## ratio = fraction of validation set size wrt. total dataset size
    ##
    ## Output:
    ## training and validation datasets

    num_ts = len(X)           # number of time series in X
    data = []

    Y = np.toarray(Y)         # labels for each time series
    # get unique class labels in Y
    unique_labels = np.unique(Y)
    num_unique_labels = Y.shape[0]  # number of unique class labels in time-series data
    Y_indexes = []                  # converting string labels into indices 0,1,2,....,n_classes-1
    for i in range(num_ts):
        Y_indexes.append(unique_labels.index(Y[i]))

    # generate one-hot encoding using numpy
    #hot_encoding = tf.one_hot(Y_indexes,num_unique_labels)
    hot_encoding = np.zeros((num_ts,num_unique_labels))
    hot_encoding[range(num_ts),Y_indexes] = 1

    for i in range(len(X)):
        data.append([X[i],hot_encoding[i]])           # X[i] = ith time series, Y[i] = ith series one-hot encoded label

    # random shuffle data and labels
    random.shuffle(data)

    # then break into training and validation sets
    validation_size = int(ratio * num_ts)
    training_data = data[:,-validation_size]
    validation_data = data[-validation_size:]

    return training_data, validation_data

## takes data in different format(data includes labels as well)
## ignore for now
def sort_timeseries_len(data, max_size):
    ## Input:
    ## data = training data
    ## max_size = length of longest Time Series
    ## Output:
    ## returns batches of different length time series, sorted by length

    batches = [ [ [] for i in range(max_size)], [[] for i in range(max_size)] ]
    for i in range(1,max_size+1):
        for j in range(len(data)):
            if len(data[j][0]) == i + 1:
                batches[0][i].append(data[j][0])
                batches[1][i].append(data[j][1])
    i = 0
    while i<len(batches[0]) :
        if len(batches[0][i]) == 0:
            del(batches[0][i])
            del(batches[1][i])
        i += 1

    return batches


def sort_by_length(data, max_batch_size, labels):
    ###max_batch_size is the max length of time series
    batches = [ [ [] for i in range(max_batch_size)], [[] for i in range(max_batch_size)] ]
    for i in range(1,max_batch_size+1):
        for j in range(len(data)):
            if len(data[j]) == i:
                batches[0][i-1].append(data[j])
                batches[1][i-1].append(labels[j])
    i = 0
    total = 0
    for j in range(1,max_batch_size+1):
        total+=len(batches[0][j-1])
        ###print ("ts length: %d num: %d" %(j,len(batches[0][j-1])))

    ###print ("total number of time series: %d" % total)
    while len(batches[0]) > i :
    #for i in range(len(batches[0])):
        # ###print(len(batches[0]))
        # ###print(i)
        if batches[0][i] == 0:
            del(batches[0][i])
            del(batches[1][i])
        i += 1
    ### hacky fix for small bug in data generation process
    del(batches[0][max_batch_size - 1])
    del(batches[1][max_batch_size - 1])
    ###print("in sort y length", batches[1][-1])
    total = 0
    for j in range(1,max_batch_size):
        total+=len(batches[0][j-1])
        ###print ("ts length: %d num: %d" %(j,len(batches[0][j-1])))

    ###print ("total number of time series: %d" % total)
    return batches

a, y = generate_data(2000, 6, 5)

###sort synthetic data by length
training_batches = sort_by_length(a[:1000], 5, y[:1000]) # for training
validation_batches = sort_by_length(a[1000:], 5, y[1000:]) # for validation



eval_obj = eval.EvalModel().build()



jiffy = models.jiffy.JiffyModel(  , 6, 2, 40, 10)

# train_jiffy(2, 2, training_batches, 100)
# validate_jiffy(2, 2, training_batches,validation_batches)


# train_jiffy(2, 2, training_batches, 100)
# validate_jiffy(2, 2, training_batches,validation_batches)
