"""
Loading data into correct format for pipeline.
"""

import numpy as np
import os
import scipy.io as scio
import fnmatch


###Data link: https://archive.ics.uci.edu/ml/machine-learning-databases/libras/
###No split- complete data is in movement_libras.data
def load_libras(data_dir, path_info):
    src_path = os.path.join(data_dir, path_info)

    data = np.genfromtxt(src_path, delimiter=',')

    X = data[:,:-1]
    y = data[:,-1]-1 # class labels are 1-indexed, should be 0-indexed

    X_tseries = []
    for x in X:
        x_0 = x[::2]
        x_1 = x[1::2]
        X_tseries.append(np.stack((x_0,x_1), axis=1))

    return (np.expand_dims(X_tseries, axis=3), y)


###Source path is a tuple containing both paths for training/testing
###Data link: https://archive.ics.uci.edu/ml/machine-learning-databases/00195/
###Split used- testing data: Test_Arabic_Digit.txt , training data: Train_Arabic_Digit.txt
def load_arabic_digits(data_dir, path_info):
    src_path_tr = os.path.join(data_dir, path_info[0])
    f_tr = open(src_path_tr, 'r')
    X_tr = []
    cur_ts = [] #used in loop hold cur timeseries in the iteration


    for (i, line) in enumerate(f_tr.readlines()):
        if line[0] != ' ' and (i != 0):
            no_newline_char = line[0:-1]
            cur_ts.append([float(n) for n in no_newline_char.split(" ")])
        if (line[0] == ' ') and (i != 0): #first line is empty,and no empty line after last ts
            X_tr.append(np.expand_dims(np.array(cur_ts), axis = 2))
            cur_ts = []
    X_tr.append(np.expand_dims(np.array(cur_ts), axis = 2))



    src_path_te = os.path.join(data_dir, path_info[1])
    f_tr = open(src_path_te, 'r')
    X_te = []
    cur_ts = []
    for (i, line) in enumerate(f_tr.readlines()):
        if line[0] != ' ' and (i != 0):
            no_newline_char = line[0:-1]
            cur_ts.append([float(n) for n in no_newline_char.split(" ")])
        if ( (line[0] == ' ') and (i != 0)): #first line is empty,and no empty line after last ts
            X_te.append(np.expand_dims(np.array(cur_ts), axis = 2))
            cur_ts = []
    X_te.append(np.expand_dims(np.array(cur_ts), axis = 2))

    ###First 660 are digit 0 next 660 are digit 1 etc..
    ###Make training labels
    y_tr = [int(i/660) for i in range(660*10)]

    ###Test data grouped into groups of 220
    ###Make test labels
    y_te = [int(i/220) for i in range(220*10)]

    # return (np.expand_dims(X_tr, axis =3), y_tr, np.expand_dims(X_te, axis = 3), y_te)
    return (X_tr, y_tr, X_te, y_te)

###Data link: https://archive.ics.uci.edu/ml/machine-learning-databases/character-trajectories/
###No split: complete data is in mixoutALL_shifted.mat
def load_trajectories(data_dir, path_info):
    """
    "/Users/chirag/Downloads/mixoutALL_shifted.mat"
    """
    # #### load data form character set
    src_path = os.path.join(data_dir, path_info)

    mat_contents = scio.loadmat(src_path)

    lens = []
    mixout = mat_contents['mixout']
    consts = mat_contents['consts']

    y = consts['charlabels'][0][0][0] - 1

    num_series = len(mixout[0])

    X = []
    for i in range(num_series):
        series_i = mixout[0][i].T
        series_i = np.expand_dims(series_i,axis=2)
        X.append(series_i)

    return X, y


###Data link: http://www.cs.cmu.edu/~bobski/
###No split, but data broken two directories normal, and abnormal- one for each of the two classes
def load_ecg(data_dir, path_info):
    """
    "/Users/chirag/Downloads/ecg/normal/"
    "/Users/chirag/Downloads/ecg/abnormal/"
    """
    src_dir_normal = os.path.join(data_dir, path_info[0])
    src_dir_abnormal = os.path.join(data_dir, path_info[1])

    X = []
    y = []
    files = os.listdir(src_dir_normal)
    for i in range(133):
        electrode_1 = np.genfromtxt(os.path.join(src_dir_normal, files[3*i]))
        electrode_2 = np.genfromtxt(os.path.join(src_dir_normal, files[3*i+1]))

        x_i = np.vstack((electrode_1[:,1],electrode_2[:,1])).T
        x_i = np.expand_dims(x_i,axis=3)

        X.append(x_i)
        y.append(0)

    files = os.listdir(src_dir_abnormal)
    for i in range(67):
        electrode_1 = np.genfromtxt(os.path.join(src_dir_abnormal, files[3*i]))
        electrode_2 = np.genfromtxt(os.path.join(src_dir_abnormal, files[3*i+1]))

        x_i = np.vstack((electrode_1[:, 1], electrode_2[:, 1])).T
        x_i = np.expand_dims(x_i, axis=3)

        X.append(x_i)
        y.append(1)


    return X, y


###Data link: http://www.cs.cmu.edu/~bobski/
###No split, but data broken into two directories, normal and abnormal - one for each of the two classes
def load_wafer(data_dir, path_info):
    """
    normal_folder = "/Users/chirag/Downloads/wafer/normal/"
    abnormal_folder = "/Users/chirag/Downloads/wafer/abnormal/"
    """
    src_dir_normal = os.path.join(data_dir, path_info[0])
    src_dir_abnormal = os.path.join(data_dir, path_info[1])

    X = []
    y = []
    files = os.listdir(src_dir_normal)
    files.sort()
    x_i = []
    for i in range(1067):
        sensor_1 = np.genfromtxt(os.path.join(src_dir_normal, files[7*i+0]))
        sensor_2 = np.genfromtxt(os.path.join(src_dir_normal, files[7*i+1]))
        sensor_3 = np.genfromtxt(os.path.join(src_dir_normal, files[7*i+2]))
        sensor_4 = np.genfromtxt(os.path.join(src_dir_normal, files[7*i+3]))
        sensor_5 = np.genfromtxt(os.path.join(src_dir_normal, files[7*i+4]))
        sensor_6 = np.genfromtxt(os.path.join(src_dir_normal, files[7*i+5]))

        #TODO: Crashing here (on vstack)
        x_i = np.vstack((sensor_1[:,1],sensor_2[:,1],sensor_3[:,1],sensor_4[:,1],sensor_5[:,1],sensor_6[:,1])).T
        x_i = np.expand_dims(x_i,axis=3)

        X.append(x_i)
        y.append(0)

    files = os.listdir(src_dir_abnormal)
    files.sort()
    # Remove box files
    box_files = fnmatch.filter(os.listdir(src_dir_abnormal),'*.box')
    for box in box_files:
        files.remove(box)
    try:
        files.remove('.DS_Store')
    except:
        print "no .DS_Store files found"

    for i in range(127):
        sensor_1 = np.genfromtxt(os.path.join(src_dir_abnormal, files[7*i]))
        sensor_2 = np.genfromtxt(os.path.join(src_dir_abnormal, files[7*i+1]))
        sensor_3 = np.genfromtxt(os.path.join(src_dir_abnormal, files[7*i+2]))
        sensor_4 = np.genfromtxt(os.path.join(src_dir_abnormal, files[7 * i + 3]))
        sensor_5 = np.genfromtxt(os.path.join(src_dir_abnormal, files[7 * i + 4]))
        sensor_6 = np.genfromtxt(os.path.join(src_dir_abnormal, files[7 * i + 5]))

        x_i = np.vstack((sensor_1[:,1],sensor_2[:,1],sensor_3[:,1],sensor_4[:,1],sensor_5[:,1],sensor_6[:,1])).T
        x_i = np.expand_dims(x_i,axis=3)

        X.append(x_i)
        y.append(1)

    return X, y

###Data link: https://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/
###No split
def load_auslan(data_dir, path_info):
    X = []
    y = []
    labels = []
    #auslan_labels = data_dir+"labels.txt"
    #labels = open(auslan_labels).read().split(",")[:25]

    src_dir = os.path.join(data_dir, path_info)
    for dirs in os.listdir(src_dir):
        src_sub_dir = os.path.join(src_dir, dirs)

        if os.path.isdir(src_sub_dir):
            for files in os.listdir(src_sub_dir):
                index = files.index('-')
                label_i = files[:index]
                labels.append(label_i)

    # Grab first 25 alphabetical (excluding capitals)
    labels = np.sort(np.unique(labels))
    labels = labels[3:28]
    print len(labels), labels
    labels_to_class = dict(zip(labels,np.arange(len(labels))))

    for dirs in os.listdir(src_dir):
        src_sub_dir = os.path.join(src_dir, dirs)

        if os.path.isdir(src_sub_dir):
            for file in os.listdir(src_sub_dir):
                index = file.index('-')
                label_i = file[:index]
                if label_i in labels:
                    src_path_data = os.path.join(src_sub_dir, file)

                    data = np.genfromtxt(src_path_data)
                    data = np.expand_dims(data,axis=3)

                    X.append(data)
                    y.append(labels_to_class[label_i])

    return X, y

#############################################################################
###Begin UCR datasets
###All files appear to be in the same csv format.
###All datasets have a training/testing split. Testing data: *dataset_name_here*_TEST, Training data: *dataset_name_here*_TRAIN
###Link to UCR repo: http://www.cs.ucr.edu/~eamonn/time_series_data/
###Download zip file, and use password: attempttoclassify

###General loading function- for univariate series from UCR only need to use this
def load_uni_csv(data_dir, path_info):
    #get paths
    src_path_tr = os.path.join(data_dir, path_info[0])
    src_path_te = os.path.join(data_dir, path_info[1])
    #load data
    data_tr = np.genfromtxt(src_path_tr, delimiter=',')
    data_te = np.genfromtxt(src_path_te, delimiter=',')
    #separate into training and testing
    X_tr = data_tr[:, 1:]
    y_tr = data_tr[:, 0]
    X_te = data_te[:, 1:]
    y_te = data_te[:, 0]

    # Make sure labels start at 0
    if np.min(y_tr) == -1:
        y_tr -= np.min(y_tr)
        y_te -= np.min(y_te)

        y_tr /= np.max(y_tr)
        y_te /= np.max(y_te)

    y_tr -= np.min(y_tr)
    y_te -= np.min(y_te)

    #reshape
    shape_tr = np.shape(X_tr)
    shape_te = np.shape(X_te)

    X_tr = np.reshape(X_tr, [shape_tr[0], shape_tr[1], 1, 1])
    X_te = np.reshape(X_te, [shape_te[0], shape_te[1], 1, 1])
    return X_tr, y_tr, X_te, y_te
