"""
    This is a deepNN
"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import csv
import timeit


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

start = timeit.default_timer()

# this is for data_stlf.csv
# X_train = np.zeros((6887, 19))
# y_train = np.zeros((6887, 1))
# X_test = np.zeros((1705, 19))
# y_test = np.zeros((1705, 1))
# count = 0

# for week.csv
# X_train = np.zeros((4915, 19))
# y_train = np.zeros((4915, 1))
# X_test = np.zeros((1229, 19))
# y_test = np.zeros((1229, 1))
# count = 0

X_train = np.zeros((1958, 19))
y_train = np.zeros((1958, 1))
X_test = np.zeros((490, 19))
y_test = np.zeros((490, 1))
count = 0

# read the data
with open('weekend.csv', 'rb') as f:
    line = csv.reader(f)
    for row in line:
        col = 0
        for data in row:
            if count >= 1958:
                if col == 0:
                    y_test[count - 1958][col] = data
                    col += 1
                else:
                    X_test[count - 1959][col - 1] = data
                    col += 1
            else:
                if col == 0:
                    y_train[count][col] = data
                    col += 1
                else:
                    X_train[count - 1][col - 1] = data
                    col += 1
        count += 1
print "finish data"

# The model
model = Sequential()
model.add(Dense(input_dim=19, output_dim=50, init='uniform'))
model.add(Activation('sigmoid'))
# model.add(Dropout(0.2))
model.add(Dense(input_dim=50, output_dim=60, init='uniform'))
model.add(Activation('sigmoid'))
# model.add(Dropout(0.2))
model.add(Dense(input_dim=60, output_dim=80, init='uniform'))
model.add(Activation('sigmoid'))
# model.add(Dropout(0.2))
model.add(Dense(input_dim=80, output_dim=60, init='uniform'))
model.add(Activation('sigmoid'))
# # model.add(Dropout(0.2))
model.add(Dense(input_dim=60, output_dim=40, init='uniform'))
model.add(Activation('sigmoid'))
# model.add(Dropout(0.2))
model.add(Dense(input_dim=40, output_dim=1, init='uniform'))
# model.add(Activation('linear'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='rmsprop')


model.fit(X_train, y_train, nb_epoch=400, verbose=0, validation_split=0.15)
print "finish model"

# Do prediction and save as csv
csv_data = np.zeros((490, 1))
for i in range(490):
    a = X_test[i]
    score = model.predict(np.reshape(a, (1, 19)))
    csv_data[i][0] = score
    # csv_data[i][1] = y_test[i]

# np.savetxt("results/weekend/deepNN_3_t.csv", csv_data, delimiter=",")

print mean_absolute_percentage_error(y_test, csv_data)

stop = timeit.default_timer()

print "TIME"
print stop - start