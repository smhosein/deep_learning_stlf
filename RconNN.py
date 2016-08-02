"""
    This is a ConvolutionNN with LSTM
"""

from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import csv
import timeit

# set parameters:
nb_filter = 250
filter_length = 3
hidden_dims = 100
nb_epoch = 200

X_train = np.zeros((6887, 19, 1))
y_train = np.zeros((6887, 1))
X_test = np.zeros((1705, 19, 1))
y_test = np.zeros((1705, 1))
count = 0


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

start = timeit.default_timer()

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
print("finish data")

model = Sequential()
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        input_shape=X_train.shape[1:]))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(50))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

print('Train...')
model.fit(X_train, y_train, nb_epoch=nb_epoch, verbose=0, validation_split=0.15)

csv_data = model.predict(X_test)

np.savetxt("results/four/RconNN.csv", csv_data, delimiter=",")

print(mean_absolute_percentage_error(y_test, csv_data))

stop = timeit.default_timer()

print("TIME")
print(stop - start)