"""
    This is a deepNN pretrainied with an autoencoder
"""
from keras.models import Sequential
from keras.layers.core import Dense, AutoEncoder, Dropout
# from keras.layers.recurrent import GRU
import numpy as np
import csv
import timeit


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

start = timeit.default_timer()

# this is for data_stlf.csv
X_train = np.zeros((6887, 19))
y_train = np.zeros((6887, 1))
X_test = np.zeros((1705, 19))
y_test = np.zeros((1705, 1))
# count = 0

# for week.csv
# X_train = np.zeros((4915, 19))
# y_train = np.zeros((4915, 1))
# X_test = np.zeros((1229, 19))
# y_test = np.zeros((1229, 1))

# X_train = np.zeros((1958, 19))
# y_train = np.zeros((1958, 1))
# X_test = np.zeros((490, 19))
# y_test = np.zeros((490, 1))
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
                    X_test[count - 6888][col - 1] = data
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

# First autoencoder
AE_0 = Sequential()


encoder = Sequential([Dense(30, input_dim=19, activation='relu')])
decoder = Sequential([Dense(19, input_dim=30, activation='relu')])

AE_0.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))
AE_0.compile(loss='mse', optimizer='rmsprop')
AE_0.fit(X_train, X_train, nb_epoch=10)

temp = Sequential()
temp.add(encoder)
temp.compile(loss='mse', optimizer='rmsprop')

first_output = temp.predict(X_train)
print first_output.shape

# Second Autoencoder
AE_1 = Sequential()

encoder_0 = Sequential([Dense(50, input_dim=30, activation='relu')])
decoder_0 = Sequential([Dense(30, input_dim=50, activation='relu')])

AE_1.add(AutoEncoder(encoder=encoder_0, decoder=decoder_0, output_reconstruction=True))
AE_1.compile(loss='mse', optimizer='rmsprop')
AE_1.fit(first_output, first_output, nb_epoch=10)

temp2 = Sequential()
temp2.add(encoder_0)
temp2.compile(loss='mse', optimizer='rmsprop')

second_output = temp2.predict(first_output)

print second_output.shape

# Thrid Autoencoder
AE_2 = Sequential()

encoder_1 = Sequential([Dense(70, input_dim=50, activation='relu')])
decoder_1 = Sequential([Dense(50, input_dim=70, activation='relu')])

AE_2.add(AutoEncoder(encoder=encoder_1, decoder=decoder_1, output_reconstruction=True))
AE_2.compile(loss='mse', optimizer='rmsprop')
AE_2.fit(second_output, second_output, nb_epoch=10)

# temp3 = Sequential()
# temp3.add(encoder_1)
# temp3.compile(loss='mse', optimizer='rmsprop')


# third_output = temp3.predict(second_output)


# # 4th Autoencoder
# AE_3 = Sequential()

# encoder_2 = Sequential([Dense(50, input_dim=70, activation='relu')])
# decoder_2 = Sequential([Dense(70, input_dim=50, activation='relu')])

# AE_3.add(AutoEncoder(encoder=encoder_2, decoder=decoder_2, output_reconstruction=True))
# AE_3.compile(loss='mse', optimizer='rmsprop')
# AE_3.fit(third_output, third_output, nb_epoch=10)

# temp4 = Sequential()
# temp4.add(encoder_2)
# temp4.compile(loss='mse', optimizer='rmsprop')

# forth_output = temp4.predict(third_output)

# # 5th Autoencoder
# AE_4 = Sequential()

# encoder_3 = Sequential([Dense(30, input_dim=50, activation='relu')])
# decoder_3 = Sequential([Dense(50, input_dim=30, activation='relu')])

# AE_4.add(AutoEncoder(encoder=encoder_3, decoder=decoder_3, output_reconstruction=True))
# AE_4.compile(loss='mse', optimizer='rmsprop')
# AE_4.fit(forth_output, forth_output, nb_epoch=10)

# # The model
model = Sequential()

model.add(encoder)
# model.add(Dropout(0.2))
model.add(encoder_0)
# model.add(Dropout(0.2))
model.add(encoder_1)
# model.add(Dropout(0.2))
# model.add(encoder_2)
# model.add(Dropout(0.2))
# model.add(encoder_3)
# model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
# model.add(Dense(1, init='uniform'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=200, verbose=0, validation_split=0.15)
print "finish model"

# Do prediction and save as csv
csv_data = np.zeros((1705, 1))
for i in range(1705):
    a = X_test[i]
    score = model.predict(np.reshape(a, (1, 19)))
    csv_data[i][0] = score
    # csv_data[i][1] = y_test[i]

# np.savetxt("results/weekend/autoencoder_5.csv", csv_data, delimiter=",")

print mean_absolute_percentage_error(y_test, csv_data)

stop = timeit.default_timer()

print "TIME"
print stop - start