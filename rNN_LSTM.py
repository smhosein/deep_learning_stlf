"""
    This is a rNN_LSTM model
"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import csv
import timeit

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
print "finish data"

nb_classes = 1
hidden_units = 100

model = Sequential()
model.add(LSTM(70, input_shape=X_train.shape[1:]))
# model.add(LSTM(70, input_dim=2))
model.add(Dense(nb_classes))
model.add(Activation('linear'))
# rmsprop = RMSprop(lr=learning_rate)
# model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

model.compile(loss="mean_squared_error", optimizer="rmsprop")

model.fit(X_train, y_train, nb_epoch=200, validation_split=0.15, verbose=0)

csv_data = model.predict(X_test)

np.savetxt("results/two/rNN_LSTM.csv", csv_data, delimiter=",")

print mean_absolute_percentage_error(y_test, csv_data)

stop = timeit.default_timer()

print "TIME"
print stop - start
