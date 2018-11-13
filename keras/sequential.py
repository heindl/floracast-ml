from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
from sklearn import preprocessing
import pandas as pd
from keras import callbacks
from keras.utils import to_categorical
from mcfly import find_architecture

def shuffle(x, y):
    assert len(x) == len(y)
    p = np.random.permutation(len(y))
    return x[p], y[p]

def split(x, y):
    # X and Y are expected to be shuffled.
    i = int(len(y) * 0.1)
    return (x[i+1:], y[i+1:]), (x[:i], y[:i])


def read_data(filepath, number_of_classes):
    df = pd.read_csv(
        filepath,
        header=None,
        index_col=False,
    )

    to_scale = df.drop(columns=[0, 1, 2], axis=1).values

    # scaled = preprocessing.StandardScaler().fit_transform(to_scale)
    scaled = to_scale

    scaled_df = pd.concat([
        df[0].rename('label'),
        df[1].rename('ts'),
        pd.DataFrame(scaled)
    ], axis=1)

    gb = scaled_df.groupby(['ts'])

    y = []
    x = []
    for k in gb.groups:
        ex = gb.get_group(k)
        y.append(ex['label'].iloc[0])
        ex = ex.drop(['ts', 'label'], axis=1)
        x.append(ex.values)

    y = to_categorical(np.asarray(y, dtype=np.int), num_classes=number_of_classes)

    return shuffle(
        np.asarray(x, dtype=np.float64),
        y
    )

# Convolutional
# Averages the pools across the spectrum.
# Unable to get above 77% average.
def convolutional():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(720, 24)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


#LSTM
# Seems to do an excellent job. 99% percent accurracy. Probably overfitting but can include dropout.
def lstm(number_of_classes):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(720, 24)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.5))
    # model.add(LSTM(100, return_sequences=True,
    #                input_shape=(720, 12)))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(100, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # return a single vector of dimension 32
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(number_of_classes, activation='sigmoid'))

    return model


NUMBER_OF_CLASSES = 5

model = lstm(NUMBER_OF_CLASSES)

# checkpoint = callbacks.ModelCheckpoint(filepath='imdb_lstm.h5', verbose=1,
#                                        save_best_only=True)
# early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
#                                          patience=2, verbose=0, mode='auto')

# For a binary classification problem
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

x, y = read_data('/Users/m/Desktop/ParsedOccurrences/standard.csv', NUMBER_OF_CLASSES)

train, test = split(x, y)

best_model, best_params, best_model_type, knn_acc = find_architecture.find_best_architecture(
    train[0],
    train[1],
    test[0],
    test[1],
    verbose=True,
                       number_of_models=5, nr_epochs=5, subset_size=500)

print(best_model)
print(best_params)
print(best_model_type)
print(knn_acc)

# model.fit(train[0], train[1],
#           batch_size=32, epochs=15, shuffle=False, # 64 batch size
#           validation_data=(test[0], test[1]),
#           # callbacks=[checkpoint, early_stopping],
#         )

# train, test = data[:i,:], data[i:,:]

# print(test)

# y_train = y[:,0].astype(int)
# x_train = train[:,1]
#
# y_test = test[:,0].astype(int)
# x_test = test[:,1]

# model.fit(X, y, epochs=20, batch_size=32)


# score = model.evaluate(x_test, y_test, batch_size=16)

# print(model.metrics_names)
# print(score)