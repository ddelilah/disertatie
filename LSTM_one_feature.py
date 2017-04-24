# LSTM for international airline passengers problem with regression framing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import datetime
import pandas_datareader.data as web
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils import plot_model
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

def get_data():
    symbols = ['GOOGL']

    data = []
    frames = []
    for ticker in symbols:
    	partial_data = pd.read_csv("data/{}.csv".format(ticker), index_col="Date", parse_dates=True, usecols=['Date', 'Adj Close', 'Symbol'], na_values=['nan'])
    	partial_data['Percentage'] = partial_data['Adj Close'].pct_change(1)
    	partial_data = partial_data.dropna()
    	percentageDF = pd.DataFrame(partial_data)
    	dfArray = [percentageDF]
    	frames.extend(dfArray)

    return pd.concat(frames);

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
dataframe = get_data()
dataset = dataframe['Percentage'].values
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back), activation='sigmoid'))
model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
plot_model(model, to_file='model.png')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


#invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# np.savetxt("trainPredictScaled.csv", trainPredict, delimiter=",");
# np.savetxt("trainYScaled.csv", trainY, delimiter=",");
# np.savetxt("testYScaled.csv", testY, delimiter=",");
# np.savetxt("testPredictScaled.csv", testPredict, delimiter=",");

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
#shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:] = np.nan
print(trainPredict.shape)
newtrainPredictPlot = []
newtrainPredictPlot.append(trainPredictPlot[look_back:len(trainPredict)+look_back])
newtrainPredictPlot = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:] = np.nan
newTestPredictPlot = []
newTestPredictPlot.append(testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1])
newTestPredictPlot = testPredict
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1,1] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label='timeseries')
plt.plot(newtrainPredictPlot, label='train')
plt.plot(newTestPredictPlot, label='test')
plt.legend(loc='upper right')
plt.show()