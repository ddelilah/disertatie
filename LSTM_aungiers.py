import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import datetime
import pandas_datareader.data as web
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix

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

def get_data(symbols):
    data = []
    frames = []
    dataMap = {}
    for ticker in symbols:
      partial_data = pd.read_csv("data/{}.csv".format(ticker), index_col="Date", parse_dates=True, usecols=['Date','Adj Close', 'Symbol'], na_values=['nan'])
      partial_data = preprocess_data(partial_data, 'Adj Close')
      partial_data = compute_technical_indicators(partial_data, 'Logged First Difference')
      partial_data = partial_data.dropna()
    return partial_data

def construct_model(data, seq_len):

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



  training_test_data = pd.DataFrame(
      {'adj_close_history': adj_close_history,
      'EWMA50': data['EWMA50'],
      'EWMA200': data['EWMA200'], 
      'ROC20': data['ROC20'],
      'ROC125': data['ROC125']})

  #print(len(adj_close_history))
  #training_test_data['adj_close_history'] = adj_close_history.tolist()

  #training_test_data = training_test_data.dropna()

  return training_test_data

# load the dataset
symbols = ['GOOGL']
#print('data:', get_data(symbols))
construct_model(get_data(symbols), 50)
