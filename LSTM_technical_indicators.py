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

      #print(partial_data)

      # plt.figure(1)
      # plt.subplot(211)
      # plt.plot(partial_data['Logged First Difference'],lw=1, label='ACN Close Prices')
      # plt.plot(partial_data['EWMA50'],'g',lw=1, label='50-day EWMA (green)')
      # plt.plot(partial_data['EWMA200'],'r', lw=1, label='200-day EWMA (red)')
      # plt.legend(loc='upper right')
      # plt.show()

    
      # plt.subplot(212)
      # plt.plot(partial_data['Logged First Difference'],lw=1, label='ACN Close Prices Log')
      # plt.plot(partial_data['ROC20'],'g',lw=1, label='20-day ROC (green)')
      # plt.plot(partial_data['ROC125'],'r', lw=1, label='125-day ROC (red)')
      # plt.legend(loc='upper right')
      # plt.show()

    return partial_data

def construct_model(data):
    
  adj_close_1 = data['Adj Close'].shift(1)
  adj_close_2 = data['Adj Close'].shift(2)

  training_test_data = pd.DataFrame(
      {'adj_close_yesterday': adj_close_1,
      'adj_close_before_yesterday': adj_close_2,
      'EWMA50': data['EWMA50'],
      'EWMA200': data['EWMA200'], 
      'ROC20': data['ROC20'],
      'ROC125': data['ROC125']})

  training_test_data = training_test_data.dropna()

  return training_test_data

# load the dataset
symbols = ['GOOGL']
train_test_data = construct_model(get_data(symbols))  
size = train_test_data.shape[0]

# data declaration & initialization
np.random.seed(7)
history = 4
batch_size = 10
window_step = 4
input_columns_nr = 6
input_shape = tf.placeholder(tf.float32, [batch_size, input_columns_nr])
output_shape = tf.placeholder(tf.float32, [batch_size, 1])
init_state = tf.placeholder(tf.float32, [batch_size, window_step])

print(input_shape)
print(output_shape)
print(init_state)

W1 = tf.Variable(tf.truncated_normal([window_step, size]))
b1 = tf.Variable(tf.zeros([size]))

W2 = tf.Variable(tf.truncated_normal([window_step, size]))
b2 = tf.Variable(tf.zeros([size]))


# train_size = int(0.8 * size)
# train_data = train_test_data.ix[:train_size, :-1]
# train_reference = train_test_data.ix[:train_size, -1]

# #print('train data:', train_data)
# #print('train reference:', train_reference)

# test_data = train_test_data.ix[train_size:, :-1]
# test_reference = train_test_data.ix[train_size:, -1]

# model = Sequential()
# # TODO: make the model lagged data actually depend on history
# model.add(Dense(history, input_dim=history, activation='sigmoid'))
# #   model.add(Dense(int(history/2), activation='sigmoid'))
# #   #model.add(Dense(int(history/4), activation='sigmoid'))
# #   #model.add(Dense(int(history/8), activation='sigmoid'))
# #   #model.add(Dense(int(history/16), activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
# model.summary()
# #print(train_data.shape)
# #print(train_data_reference.shape)
# train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
# train_data_reference = np.reshape(train_data_reference, (train_data_reference.shape[0], 1, train_data_reference.shape[1]))

# model.fit(train_data, train_reference, shuffle=True, nb_epoch=100, batch_size=10)

# # Use the complete data set to see the network performance.
# # Regenerate data set because it was shuffled before.
# #   dataframe = get_data(symbols)
# #   train_test_data = dataframe[symbol]
# #   print('train_test_data ', train_test_data)

# test_data_predicted = model.predict(test_data)
# test_data_reference = test_data_reference

# print("test data predict: ", test_data_predicted)
# print("test data referece: ", test_data_reference)

# #   #relative_deviation = test_data_predicted/test_data_reference - 1.0
# #   #print('Relative deviation: ', relative_deviation)

# #   # calculate root mean squared error
# #   # testScore = math.sqrt(mean_squared_error(test_data_reference[0], test_data_predicted[:,0]))
# #   # print('Test Score: %.2f RMSE' % (testScore))

# #   plt.figure()
# #   plt.plot(range(len(test_data_reference)), test_data_reference, 'b-', label='reference')
# #   plt.plot(range(len(test_data_predicted)), test_data_predicted, 'r--', label='predicted')
# #   plt.xlabel('test case #')
# #   plt.ylabel('predictions')
# #   plt.title('Reference values vs predicted values')
# #   plt.legend()

# #   # plt.figure()
# #   # plt.plot(range(len(test_data_predicted)), relative_deviation, 'bx', label='relative deviation')
# #   # plt.xlabel('test case #')
# #   # plt.ylabel('relative deviation')
# #   # plt.title('Relative deviation of predicted values (predicted / reference - 1)')
# #   # plt.legend()

# #   plt.show()
