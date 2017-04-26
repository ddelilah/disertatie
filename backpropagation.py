from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import tensorflow
import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "ROC20", "ROC125", "EWMA50", "EWMA200", "Adj Close"]
FEATURES = ["Date", "Open", "High", "Low", "Close", "Volume", "ROC20", "ROC125", "EWMA50", "EWMA200"]
LABEL = "Adj Close"

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

def get_data():
    symbols = ['GOOGL']

    for ticker in symbols:
    	data = pd.read_csv("data/{}.csv".format(ticker), index_col="Date", usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], parse_dates=True, na_values=['nan'])
    	data = compute_technical_indicators(data, 'Adj Close')
    	data = data.dropna()

    return data	

def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x)))) 

def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values)
                  for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

def batch_data(example, label, batch_size, num_epochs=None):
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size
	example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)

	return example_batch, label_batch  


data = get_data()
size = data.shape[0]
train_size = int(0.8 * size)
train_test_data = data.drop('Adj Close', 1)
adj_close = data.ix[:, 5]
train_data = train_test_data.ix[:train_size, :]
train_reference = adj_close.ix[:train_size]
test_data = train_test_data.ix[train_size:, :]
#print('Train data: ', train_data)
#print('Test data: ', test_data)
feature_cols = [tf.contrib.layers.real_valued_column(k)
					for k in FEATURES]


input_shape = tf.placeholder(tf.float32, [None, 784])
output_shape = tf.placeholder(tf.float32, [None, 1])

middle = 30
w_1 = tf.Variable(tf.truncated_normal([784, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, 1]))
b_2 = tf.Variable(tf.truncated_normal([1, 1]))

z_1 = tf.add(tf.matmul(input_shape, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)

diff = tf.subtract(a_2, output_shape)

d_z_2 = tf.multiply(diff, sigmaprime(z_2))
d_b_2 = d_z_2
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
d_b_1 = d_z_1
d_w_1 = tf.matmul(tf.transpose(input_shape), d_z_1)

eta = tf.constant(0.5)
step = [
    tf.assign(w_1,
            tf.subtract(w_1, tf.multiply(eta, d_w_1)))
  , tf.assign(b_1,
            tf.subtract(b_1, tf.multiply(eta,
                               tf.reduce_mean(d_b_1, axis=[0]))))
  , tf.assign(w_2,
            tf.subtract(w_2, tf.multiply(eta, d_w_2)))
  , tf.assign(b_2,
            tf.subtract(b_2, tf.multiply(eta,
                               tf.reduce_mean(d_b_2, axis=[0]))))
]

acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(output_shape, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_data_tensor = tf.constant(train_data.as_matrix(), dtype = tf.float32, shape=[458, 10])
train_reference_tensor = tf.constant(train_reference.as_matrix(), dtype = tf.float32, shape=[train_reference.shape[0], 1])

for i in range(10000):
    batch_xs, batch_ys = batch_data(train_data_tensor, train_reference_tensor, 10)
    sess.run(step, feed_dict = {input_shape: batch_xs,
                                output_shape : batch_ys})
    # if i % 1000 == 0:
    #     res = sess.run(acct_res, feed_dict =
    #                    {a_0: mnist.test.images[:1000],
    #                     y : mnist.test.labels[:1000]})
    #     print res
