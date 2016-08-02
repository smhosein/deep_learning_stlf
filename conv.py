"""
    This is a ConvolutionNN
"""

from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import csv
import timeit


# set parameters:
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 400


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

start = timeit.default_timer()

X_train = np.zeros((6887, 19, 1))
y_train = np.zeros((6887, 1))
X_test = np.zeros((1705, 19, 1))
y_test = np.zeros((1705, 1))
count = 0

# read the data
with open('data_stlf.csv', 'rb') as f:
    line = csv.reader(f)
    for row in line:
        col = 0
        for data in row:
            if count >= 6887:
                if col == 0:
                    y_test[count - 6887][col] = data
                    col += 1
                else:
                    X_test[count - 6888][col - 1][0] = data
                    col += 1
            else:
                if col == 0:
                    y_train[count][col] = data
                    col += 1
                else:
                    X_train[count - 1][col - 1][0] = data
                    col += 1
        count += 1
print ("finish data")


print('Build model...')
model = Sequential()

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        input_shape=X_train.shape[1:]))
# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=2))

# model.add(Convolution1D(nb_filter=60,
#                         filter_length=filter_length,
#                         border_mode='valid',
#                         activation='relu',
#                         input_shape=X_train.shape[1:]))
# # we use standard max pooling (halving the output of the previous layer):
# model.add(MaxPooling1D(pool_length=2))
# # We flatten the output of the conv layer,
# # so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
# model.add(Dropout(0.25))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(X_train, y_train, nb_epoch=nb_epoch, verbose=0, validation_split=0.15)

csv_data = model.predict(X_test)

np.savetxt("results/four/conv_t.csv", csv_data, delimiter=",")

print(mean_absolute_percentage_error(y_test, csv_data))

stop = timeit.default_timer()

print("TIME")
print(stop - start)