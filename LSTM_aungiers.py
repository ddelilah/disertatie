import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import time
import pandas_datareader.data as web
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix
from keras.utils import plot_model

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

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

#-------------- Data pre-processing (in order to make a stationary time-series) -------------------------------  

def preprocess_data(data, feature):
  data['Logged First Difference'] = np.log(data[feature]) - np.log(data[feature].shift())

  return data

def get_data(symbols, seq_len):
    data = []
    frames = []
    dataMap = {}
    for ticker in symbols:
      partial_data = pd.read_csv("data/{}.csv".format(ticker), index_col="Date", parse_dates=True, usecols=['Date','Adj Close'], na_values=['nan'])
      #partial_data = preprocess_data(partial_data, 'Adj Close')
      partial_data = compute_technical_indicators(partial_data, 'Adj Close')
      partial_data = partial_data.dropna()
      data = partial_data

    sequence_length = seq_len + 1
    adj_close_history = []
    ewma50_history = []
    ewma200_history = []
    roc20_history = []
    roc125_history = []

    for index in range(len(data) - sequence_length):
    	adj_close_history.append(data['Adj Close'][index: index + sequence_length])
    	ewma50_history.append(data['EWMA50'][index: index + sequence_length])
    	ewma200_history.append(data['EWMA200'][index: index + sequence_length])
    	roc20_history.append(data['ROC20'][index: index + sequence_length])
    	roc125_history.append(data['ROC125'][index: index + sequence_length])

    adj_close_history = np.array(adj_close_history)
    ewma50_history = np.array(ewma50_history)
    ewma200_history = np.array(ewma200_history)
    roc20_history = np.array(roc20_history)
    roc125_history = np.array(roc125_history)

    training_test_data = np.array([
    	adj_close_history,
    	ewma50_history,
    	ewma200_history,
    	roc20_history,
    	roc125_history])

    size = training_test_data.shape[0]
    train_size = int(0.8 * size)
    train_data = training_test_data[:train_size, :]
    np.random.shuffle(train_data)
    x_train = train_data[:, :-1]
    
    #train reference
    y_train = train_data[:, -1]
    x_test = training_test_data[train_size:, :-1]
    y_test = training_test_data[train_size:, -1]

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 51)) 
    y_test = np.reshape(y_test, (y_test.shape[1]))
    return [x_train, y_train, x_test, y_test]

def build_model(layers):
	model = Sequential()

	model.add(LSTM(input_shape=(69, 51),
		output_dim=layers[1],
		return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(layers[2],
		return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(output_dim=layers[3]))
	model.add(Activation("linear"))
	start = time.time()
	model.compile(loss="mse", optimizer="rmsprop")
	print("> Compilation Time : ", time.time() - start)

	#plot_model(model, to_file='model.png')
	#print(model.summary())
	return model

def predict_point_by_point(model, data):
	#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
	predicted = model.predict(data)
	predicted = np.reshape(predicted, (predicted.size,))
	return predicted

def plot_results(predicted_data, true_data):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.show()

symbols = ['GLD']
epochs = 50
seq_len = 50
print('> Loading data... ')
X_train, y_train, X_test, y_test = get_data(symbols, seq_len)
print('> Data Loaded. Compiling...')
model = build_model([1, 50, 100, 51])
model.fit(X_train, y_train, batch_size=32, epochs=epochs,
	validation_split=0.05)

predicted = predict_point_by_point(model, X_test)
plot_results(predicted, y_test)
print('predicted:', predicted)
print('y_test', y_test)