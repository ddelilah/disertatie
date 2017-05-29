import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import time
import pandas_datareader.data as web
from keras.models import Sequential
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix
from keras.utils import plot_model

seq_len = 50

# ----------  Technical indicators -------------------
def compute_EWMA(data, ndays, feature):
  EMA = pd.Series(pd.ewma(data[feature], span = ndays, min_periods = ndays - 1), name = 'EWMA_' + str(ndays)) 
  return data.join(EMA) 

def compute_technical_indicators(data, feature):
  #compute 20-day Rate-of-change
  data['ROC20'] = data[feature].pct_change(20)
  #compute 125-day Rate-of-change
  data['ROC125'] = data[feature].pct_change(125)

  #compute 50 days EWMA
  EWMA50 = compute_EWMA(data, 50, feature)
  data['EWMA50'] = EWMA50['EWMA_50']
  

  #compute 200 days EWMA
  EWMA200 = compute_EWMA(data, 200, feature)
  data['EWMA200'] = EWMA200['EWMA_200']

  return data

#-------------- Data pre-processing (in order to make a stationary time-series) -------------------------------  

def preprocess_data(data, feature):
  data['Logged First Difference'] = np.log(data[feature]) - np.log(data[feature].shift())

  return data

def normalise_windows(window_data):
  normalised_data = []
  for window in window_data:
    #normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
    #normalised_data.append(normalised_window)
    #print('normalized value:', (window - np.mean(window))/np.std(window))
    normalised_data.append((window - np.mean(window))/np.std(window))
  return normalised_data

def get_data(symbols, seq_len):
    data = []
    frames = []
    dataMap = {}
    for ticker in symbols:
      partial_data = pd.read_csv("data/{}.csv".format(ticker), index_col="Date", parse_dates=True, usecols=['Date','Adj Close'], na_values=['nan'])
      partial_data = compute_technical_indicators(partial_data, 'Adj Close')
      partial_data = partial_data.dropna()
      data = partial_data.iloc[::-1]

    sequence_length = seq_len + 1
    adj_close_history = []
    ewma50_history = []
    ewma200_history = []
    roc20_history = []
    roc125_history = []

    print(len(data) - sequence_length)
    for index in range(len(data) - sequence_length):
    	# use iloc[::-1] to reverse the order (last entry in the array should be the value of the current timestep)
      adj_close_history.append(data['Adj Close'][index: index + sequence_length].iloc[::-1])
      ewma50_history.append(data['EWMA50'][index: index + sequence_length].iloc[::-1])
      ewma200_history.append(data['EWMA200'][index: index + sequence_length].iloc[::-1])
      roc20_history.append(data['ROC20'][index: index + sequence_length].iloc[::-1])
      roc125_history.append(data['ROC125'][index: index + sequence_length].iloc[::-1])

    adj_close_history = normalise_windows(adj_close_history)
    adj_close_history = np.array(adj_close_history)

    ewma50_history = normalise_windows(ewma50_history)
    ewma50_history = np.array(ewma50_history)

    ewma200_history = normalise_windows(ewma200_history)
    ewma200_history = np.array(ewma200_history)

    #roc20_history = normalise_windows(roc20_history)
    roc20_history = np.array(roc20_history)

    roc125_history = normalise_windows(roc125_history)
    roc125_history = np.array(roc125_history)

    input_size = len(adj_close_history)

    timesteps = [ [] for _ in range(input_size) ]

    for i in range(input_size):
      timesteps[i].append(adj_close_history[i])
      timesteps[i].append(ewma50_history[i])
      #timesteps[i].append(ewma200_history[i])
      timesteps[i].append(roc20_history[i])
      timesteps[i].append(roc125_history[i])

    training_test_data = np.array(timesteps)
    print(timesteps[0])
    print('training_test_data:', training_test_data)
    print('shape:', training_test_data.shape)
    #plt.plot(training_test_data, label='training_test_data')
    #plt.legend()
    #plt.show()

    size = training_test_data.shape[0]
    train_size = int(0.8 * size)
    train_data = training_test_data[:train_size, :]
    print(train_data.shape)
    np.random.shuffle(train_data)
    print('train_data:', train_data)
    x_train = train_data[:, :, :-1]
    print('x_train:', x_train)
    print('shape:', x_train.shape)
    #plt.plot(x_train, label='x_train')
    #plt.legend()
    #plt.show()

    
    #train reference
    y_train = train_data[:, :, -1]
    print('y_train:', y_train)
    print('shape:', y_train.shape)
    #plt.plot(y_train, label='y_train')
    #plt.legend()
    #plt.show()

    print('train_size: ', train_size)
    x_test = training_test_data[train_size:, :, :-1]
    print('x_test before:', x_test)
    print('x_test shape:', x_test.shape)

    y_test = training_test_data[train_size:, :, -1]
    print('y_test before:', y_test)
    print('y_test shape:', y_test.shape)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 50))
    #print('x_train after reshape:', x_train)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 50))
    #print('x_test after reshape:', x_test)

    return [x_train, y_train, x_test, y_test]

def build_model(layers):
	model = Sequential()

	model.add(LSTM(input_shape=(4, 50),
		output_dim=layers[0],
		return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(layers[1],
		return_sequences=False))
	model.add(Dropout(0.2))

	model.add(Dense(output_dim=4))
	model.add(Activation("linear"))
	#model.add(Activation("sigmoid"))
	start = time.time()
	model.compile(loss="mse", optimizer="adam")
	print("> Compilation Time : ", time.time() - start)

	#plot_model(model, to_file='model.png')
	#print(model.summary())
	return model

def predict_point_by_point(model, data):
	#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
	predicted = model.predict(data)
	print('predicted in pbp:', predicted.shape)
	#predicted = np.reshape(predicted, (predicted.size,))
	return predicted

def plot_results(predicted_data, true_data):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.show()

symbols = ['IBM']
epochs = 100
print('> Loading data... ')
X_train, y_train, X_test, y_test = get_data(symbols, seq_len)
#y_train = np.reshape(y_train, (y_train.shape[1], y_train.shape[0], y_train.shape[2]))
print('ff:', y_train.shape)
print('ddd:', X_train.shape)
print('X_train: ', X_train)
print('y_train: ', y_train)
print('X_test: ', X_test)
print('y_test: ', y_test)
print('> Data Loaded. Compiling...')
model = build_model([50, 64, 64, 1])
model.summary()
model.fit(X_train, y_train, batch_size=50, epochs=epochs, validation_split=0.1)
predicted = predict_point_by_point(model, X_test)
plot_results(predicted, y_test)
print('predicted:', predicted)
print('y_test', y_test)