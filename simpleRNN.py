"""
    This is a SimpleRNN
"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
import numpy as np
import csv
import timeit


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
print "finish data"

nb_classes = 1
hidden_units = 100

model = Sequential()
model.add(SimpleRNN(70, activation='relu', input_shape=X_train.shape[1:]))
model.add(Dense(nb_classes))
model.add(Activation('linear'))

# model.add(LSTM(5, 300, return_sequences=True))
# model.add(LSTM(300, 500, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(500, 200, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(200, 3))

model.compile(loss="mean_squared_error", optimizer="rmsprop")

model.fit(X_train, y_train, nb_epoch=200, validation_split=0.15, verbose=0)

csv_data = model.predict(X_test)

# np.savetxt("results/four/SimpleRNN.csv", csv_data, delimiter=",")

print mean_absolute_percentage_error(y_test, csv_data)

stop = timeit.default_timer()

print "TIME"
print stop - start
