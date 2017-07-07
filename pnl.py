from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import random
from keras.models import Sequential
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import StandardScaler
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils import plot_model

seq_len = 50
scaler = StandardScaler()
scalery = StandardScaler()


# ----------  Technical indicators -------------------
def compute_EWMA(data, ndays, feature):
    EMA = pd.Series(pd.ewma(data[feature], span=ndays, min_periods=ndays - 1), name='EWMA_' + str(ndays))
    return data.join(EMA)


def compute_technical_indicators(data, feature):
    # compute 20-day Rate-of-change
    data['ROC50'] = data[feature].pct_change(50)
    # compute 125-day Rate-of-change
    data['ROC125'] = data[feature].pct_change(125)

    # compute 50 days EWMA
    EWMA50 = compute_EWMA(data, 50, feature)
    data['EWMA50'] = EWMA50['EWMA_50']

    # compute 200 days EWMA
    EWMA200 = compute_EWMA(data, 200, feature)
    data['EWMA200'] = EWMA200['EWMA_200']

    return data

def get_data(symbols, seq_len):
    data = []
    frames = []
    dataMap = {}
    for ticker in symbols:
        partial_data = pd.read_csv("data/{}.csv".format(ticker), index_col="Date", parse_dates=True,
                                   usecols=['Date', 'Adj Close', 'Open'], na_values=['nan'])
        # exchange_data = pd.read_csv("data/exchange.csv", index_col="Date", parse_dates=True,
        #                            usecols=['Date', 'EUR/USD Close'], na_values=['nan'])
        #partial_data = compute_technical_indicators(partial_data, 'Adj Close')
        #data = pd.concat([partial_data, spy_data], axis=1).iloc[::-1]
        #data = data.dropna()
        partial_data = partial_data.dropna()
        data = partial_data.iloc[::-1]

    print('data:', data)

    sequence_length = seq_len + 1
    adj_close_history = []
    open_history = []

    for index in range(len(data) - sequence_length):
        # use iloc[::-1] to reverse the order (last entry in the array should be the value of the current timestep)
        adj_close_history.append(data['Adj Close'][index: index + sequence_length].iloc[::-1])
        open_history.append(data['Open'][index: index + sequence_length].iloc[::-1])

    adj_close_history = np.array(adj_close_history)
    ewma50_history = np.array(open_history)
    print('ewma_50:', ewma50_history)

    input_size = len(adj_close_history)

    timesteps = [[] for _ in range(input_size)]

    for i in range(input_size):
        timesteps[i].append(adj_close_history[input_size - i - 1])
        timesteps[i].append(ewma50_history[input_size - i - 1])

    training_test_data = np.array(timesteps)
    print('training_test_data:', training_test_data)
    #print('shape:', training_test_data.shape)
    # plt.plot(training_test_data, label='training_test_data')
    # plt.legend()
    # plt.show()

    size = training_test_data.shape[0]
    print('size:', size)
    train_size = int(0.8 * size)
    train_data = training_test_data[:train_size, :]
    np.random.shuffle(train_data)
    # print('train_data:', train_data)
    x_train = train_data[:, :, :-1]
    #print('x_train:', x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], 2 * x_train.shape[2]))
    #print('shape:', x_train.shape)
    #print('x_train: ', x_train)
    scalerX = StandardScaler().fit(x_train)
    x_train = scalerX.transform(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

    # train reference
    y_train = train_data[:, :, -1]
    #print('y_train: ', y_train)
    global scalery
    scalery = scalery.fit(y_train)
    y_train = scalery.transform(y_train)
    #y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
    # print('y_train:', y_train)
    print('shape:', y_train.shape)
    # plt.plot(y_train, label='y_train')
    # plt.legend()
    # plt.show()

    # print('train_size: ', train_size)
    x_test = training_test_data[train_size:, :, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], 2 * x_test.shape[2]))
    x_test = scalerX.transform(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    print('x_test before:', x_test)
    # print('x_test shape:', x_test.shape)

    y_test = training_test_data[train_size:, :, -1]
    print('y_test:', y_test)
    y_test = scalery.transform(y_test)
    # print('y_test shape:', y_test.shape)

    x_train = np.reshape(x_train, (x_train.shape[0], 50, 2 * x_train.shape[1]))
    #x_train = np.reshape(x_train, (x_train.shape[0], 50))
    #print('x_train after reshape:', x_train.shape)
    x_test = np.reshape(x_test, (x_test.shape[0], 50, 2 * x_test.shape[1]))
    # print('x_test after reshape:', x_test)

    return [x_train, y_train, x_test, y_test]

# Convolution
kernel_size = 3
filters = 64
pool_size = 2

def build_model(layers):
    model = Sequential()
    model.add(Conv1D(filters,
                     kernel_size,
                     input_shape=(50, 2),
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))

    model.add(LSTM(layers[0], input_shape=(50, 2), return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(layers[1], return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=2))
    model.add(Activation("linear"))
    #model.add(BatchNormalization())
    start = time.time()
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)

    # plot_model(model, to_file='model.png')
    # print(model.summary())
    return model


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    # predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def plot_results(predicted_data, true_data, title_label):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Real data')
    plt.title(title_label)
    plt.plot(predicted_data, label='Predicted data')
    plt.xlabel('Day')
    plt.ylabel('Stock price')
    plt.legend()
    plt.show()


# fix random seed for reproducibility
np.random.seed(7)
symbols = ['FB']
epochs = 50
print('> Loading data... ')
X_train, y_train, X_test, y_test = get_data(symbols, seq_len)
print('> Data Loaded. Compiling...')
print('X_test shape:', X_test.shape)
model = build_model([128, 86])
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_split=0.1, callbacks=[early_stopping])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predicted = predict_point_by_point(model, X_test)
predicted_inversed = scalery.inverse_transform(predicted)
print('predicted_inversed:', predicted_inversed)
y_test_inversed = scalery.inverse_transform(y_test)
print('y_test_inversed:', y_test_inversed)
plot_results(predicted_inversed[:,0], y_test_inversed[:,0], 'Adj Close')
adj_error_scaled = y_test - predicted
plt.title('Adj close error')
plt.plot(adj_error_scaled[:,0], label='Prediction error (scaled)')
plt.xlabel('Day')
plt.ylabel('Prediction error')
plt.show()
score = model.evaluate(X_test, y_test, batch_size=50)
adj_error = y_test_inversed - predicted_inversed
plt.title('Adj close error')
plt.plot(adj_error[:,0], label='Prediction error (not scaled)')
plt.xlabel('Day')
plt.ylabel('Prediction error')
plt.show()
price_day_0 = predicted_inversed[0][0]
nr_stocks = 100
eur_usd_start_interval = 1.115979
eur_usd_end_interval = 1.270401

pnl = []
for i in range(len(predicted_inversed[:,0])):
    random_seed = random.uniform(eur_usd_start_interval, eur_usd_end_interval)
    pnl_value = predicted_inversed[i,0] * nr_stocks * random_seed - price_day_0 * nr_stocks * random_seed
    pnl.append(pnl_value)

plt.title('Profits and loss')
plt.plot(pnl, label='PNL')
plt.xlabel('Day')
plt.ylabel('Profit and loss value')
plt.show()
