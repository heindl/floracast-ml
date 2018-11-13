from keras.layers import GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, Input, Flatten
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.layers.merge import concatenate
from keras.models import Model

def split(x, y):
    # X and Y are expected to be shuffled.
    i = int(len(y) * 0.1)
    x_test = []
    x_train = []
    for c in x:
        x_train.append(c[i+1:])
        x_test.append(c[:i])

    x_test = np.asarray(x_test, dtype=np.float64)
    x_train = np.asarray(x_train, dtype=np.float64)

    return x_train, y[i+1:], x_test, y[:i]


def read_data(filepath):
    df = pd.read_csv(
        filepath,
        header=None,
        index_col=False,
    )

    to_scale = df.drop(columns=[0, 1, 2], axis=1).values

    scaled = preprocessing.StandardScaler().fit_transform(to_scale)

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
        for i in range(len(ex.columns)-2):
            x.append([])
        break

    for k in gb.groups:
        ex = gb.get_group(k)
        y.append(ex['label'].iloc[0])
        ex = ex.drop(['ts', 'label'], axis=1)
        for i, c in enumerate(ex.columns):
            x[i].append(
                np.expand_dims(
                    np.asarray(ex[c], dtype=np.float64),
                    axis=1
                )
            )

    # for k in gb.groups:
    #     ex = gb.get_group(k)
    #     y.append(ex['label'].iloc[0])
    #     ex = ex.drop(['ts', 'label'], axis=1)
    #     inner = []
    #     for i, c in enumerate(ex.columns):
    #         inner.append(np.expand_dims(np.asarray(ex[c], dtype=np.float64), axis=1))
    #     x.append(inner)

    x = np.asarray(x, dtype=np.float64)

    number_of_classes = len(np.unique(y))

    y = to_categorical(np.asarray(y, dtype=np.int), num_classes=number_of_classes)

    # Shuffle
    p = np.random.permutation(len(y))
    return np.asarray(map(lambda a: a[p], x)), y[p], number_of_classes
    # return x[p], y[p], number_of_classes

# Convolutional
def define_model(features, number_of_classes):

    sequence_inputs = []
    sequence_layers = []

    for i, column in enumerate(features):

        input = Input(shape=(len(column[0]), 1))
        conva1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input)
        conva2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conva1)
        pool1 = MaxPooling1D(pool_size=2)(conva2)
        # convb1 = Conv1D(filters=4, kernel_size=5, activation='relu')(pool1)
        # convb2 = Conv1D(filters=4, kernel_size=5, activation='relu')(convb1)
        # pool2 = MaxPooling1D(pool_size=2)(convb2)
        # print('pooled', pool1._keras_shape)
        # drop1 = Dropout(0.5)(conv1)
        flat1 = Flatten()(pool1)
        # print('flattened', flat1._keras_shape)
        sequence_inputs.append(input)
        sequence_layers.append(flat1)

    merged = concatenate(sequence_layers)
    # print(merged._keras_shape)
    dense2 = Dense(256, activation='relu')(merged)
    outputs = Dense(number_of_classes, activation='sigmoid')(dense2)
    model = Model(inputs=sequence_inputs, outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# checkpoint = callbacks.ModelCheckpoint(filepath='imdb_lstm.h5', verbose=1,
#                                        save_best_only=True)
# early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
#                                          patience=2, verbose=0, mode='auto')

x, y, number_of_classes = read_data('/Users/m/Desktop/ParsedOccurrences/standard.csv')

train_x, train_y, test_x, test_y = split(x, y)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

model = define_model(train_x, number_of_classes)

model.fit(x=[a for a in train_x], y=train_y,
          batch_size=32, epochs=5, shuffle=False, # 64 batch size
          validation_data=([a for a in test_x], test_y),
          # callbacks=[checkpoint, early_stopping],
        )