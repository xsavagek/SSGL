import tensorflow as tf
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
sns.set() # 使用seaborn的默认设置进行绘图
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
    ):
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell() for _ in range(num_layers)], state_is_tuple = False
        )
        self.X = tf.placeholder(tf.float32, [None, None, size])
        self.Y = tf.placeholder(tf.float32, [None, output_size])
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        drop = tf.contrib.rnn.DropoutWrapper(
            self.rnn_cells, output_keep_prob = forget_bias
        )
        _, last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        with tf.variable_scope('decoder', reuse = False):
            self.rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(num_layers)], state_is_tuple = False
            )
            drop_dec = tf.contrib.rnn.DropoutWrapper(
                self.rnn_cells_dec, output_keep_prob = forget_bias
            )
            self.outputs, self.last_state = tf.nn.dynamic_rnn(
                drop_dec, self.X, initial_state = last_state, dtype = tf.float32
            )
        self.logits = tf.layers.dense(self.outputs[:, -1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits)) ##################
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)


# 读取数据
# df = pd.read_csv('./GOOG-year.csv')
df = pd.read_csv('./dataset/IBMclose.csv')
date_ori = pd.to_datetime(df.iloc[:, 0]).tolist() #取第0列，转换日期为"2016-11-02"格式
print(df.head())

# 数据归一化
minmax = MinMaxScaler().fit(df.iloc[:, 2:-1].astype('float32')) #归一化，原数据-最大/最大-最小
df_log = minmax.transform(df.iloc[:, 2:-1].astype('float32')) #数据集中'Adj close'为修正后的收盘价
df_log = pd.DataFrame(df_log)
print("df_log.shape: ", df_log.shape) # 2513*5
print(df_log.head())
# print(df_log) #GOOG-year.csv: 252X6
# print(df_log.shape[1]) #GOOG-year.csv: 6


# 超参数
learning_rate = 0.002
num_layers = 1
size_layer = 128
timestamp = 5
epoch = 50 # 500                                                                                                                                                     
dropout_rate = 0.7
future_day = 50


tf.reset_default_graph()
modelnn = Model(learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


for i in range(epoch):
    init_value = np.zeros((1, num_layers * 2 * size_layer))
    total_loss = 0
    for k in range(0, df_log.shape[0] - 1, timestamp):
        index = min(k + timestamp, df_log.shape[0] - 1)
        batch_x = np.expand_dims(df_log.iloc[k:index].values, axis = 0)
        batch_y = df_log.iloc[k + 1 : index + 1].values
        last_state, _, loss = sess.run(
            [modelnn.last_state, modelnn.optimizer, modelnn.cost],
            feed_dict = {
                modelnn.X: batch_x,
                modelnn.Y: batch_y,
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        total_loss += loss
    total_loss /= df_log.shape[0] // timestamp
    print('epoch:', i + 1, 'avg loss:', total_loss)



output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
output_predict[0] = df_log.iloc[0] # 日期
upper_b = (df_log.shape[0] // timestamp) * timestamp
init_value = np.zeros((1, num_layers * 2 * size_layer))
for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(
                df_log.iloc[k : k + timestamp], axis = 0
            ),
            modelnn.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[k + 1 : k + timestamp + 1] = out_logits

out_logits, last_state = sess.run(
    [modelnn.logits, modelnn.last_state],
    feed_dict = {
        modelnn.X: np.expand_dims(df_log.iloc[upper_b:], axis = 0),
        modelnn.hidden_layer: init_value,
    },
)
init_value = last_state
output_predict[upper_b + 1 : df_log.shape[0] + 1] = out_logits
df_log.loc[df_log.shape[0]] = out_logits[-1]
date_ori.append(date_ori[-1] + timedelta(days = 1))


for i in range(future_day - 1):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(df_log.iloc[-timestamp:], axis = 0),
            modelnn.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[df_log.shape[0]] = out_logits[-1]
    df_log.loc[df_log.shape[0]] = out_logits[-1]
    date_ori.append(date_ori[-1] + timedelta(days = 1))


df_log = minmax.inverse_transform(output_predict)
date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()


def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer


# 衡量指标
def get_mape(y_true, y_pred): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_mae(a, b):
    """
    Comp mean absolute error e_t = E[|a_t - b_t|]. a and b can be lists.
    Returns a vector of len = len(a) = len(b)
    """
    return np.mean(abs(np.array(a)-np.array(b)))

def get_rmse(a, b):
    """
    Comp RMSE. a and b can be lists.
    Returns a scalar.
    """
    return math.sqrt(np.mean((np.array(a)-np.array(b))**2))

arr_true_close = df.iloc[:, 5]
arr_pre_close = anchor(df_log[: -50, 3], 1)
np.savetxt('results/lstm-seq2seq/IBM/arr_true_close.csv', arr_true_close, delimiter = ',')
np.savetxt('results/lstm-seq2seq/IBM/arr_pre_close.csv', arr_pre_close, delimiter = ',')


print("RMSE = %0.4f" % get_rmse(arr_true_close, arr_pre_close))
print("MAPE = %0.4f%%" % get_mape(arr_true_close, arr_pre_close))
print("MAE = %0.4f" % get_mae(arr_true_close, arr_pre_close))

indicator = np.array([get_rmse(arr_true_close, arr_pre_close), 
                      get_mape(arr_true_close, arr_pre_close), 
                      get_mae(arr_true_close, arr_pre_close)])
np.savetxt('results/lstm-seq2seq/IBM/indicator.csv', indicator, delimiter = ',')



#
#current_palette = sns.color_palette('Paired', 12)
#fig = plt.figure(figsize = (15, 10))
#ax = plt.subplot(111)
#x_range_original = np.arange(df.shape[0])
#x_range_future = np.arange(df_log.shape[0])
#ax.plot(
#    x_range_original,
#    df.iloc[:, 1],
#    label = 'true Open',
#    color = current_palette[0],
#)
#ax.plot(
#    x_range_future,
#    anchor(df_log[:, 0], 0.5),
#    label = 'predict Open',
#    color = current_palette[1],
#)
#ax.plot(
#    x_range_original,
#    df.iloc[:, 2],
#    label = 'true High',
#    color = current_palette[2],
#)
#ax.plot(
#    x_range_future,
#    anchor(df_log[:, 1], 0.5),
#    label = 'predict High',
#    color = current_palette[3],
#)
#ax.plot(
#    x_range_original,
#    df.iloc[:, 3],
#    label = 'true Low',
#    color = current_palette[4],
#)
#ax.plot(
#    x_range_future,
#    anchor(df_log[:, 2], 0.5),
#    label = 'predict Low',
#    color = current_palette[5],
#)
#ax.plot(
#    x_range_original,
#    df.iloc[:, 4],
#    label = 'true Close',
#    color = current_palette[6],
#)
#ax.plot(
#    x_range_future,
#    anchor(df_log[:, 3], 0.5),
#    label = 'predict Close',
#    color = current_palette[7],
#)
#ax.plot(
#    x_range_original,
#    df.iloc[:, 5],
#    label = 'true Adj Close',
#    color = current_palette[8],
#)
#ax.plot(
#    x_range_future,
#    anchor(df_log[:, 4], 0.5),
#    label = 'predict Adj Close',
#    color = current_palette[9],
#)
#box = ax.get_position()
#ax.set_position(
#    [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
#)
#ax.legend(
#    loc = 'upper center',
#    bbox_to_anchor = (0.5, -0.05),
#    fancybox = True,
#    shadow = True,
#    ncol = 5,
#)
#plt.title('overlap stock market')
#plt.xticks(x_range_future[::30], date_ori[::30])
#plt.show()
#
#
#fig = plt.figure(figsize = (20, 8))
#plt.subplot(1, 2, 1)
#plt.plot(
#    x_range_original,
#    df.iloc[:, 1],
#    label = 'true Open',
#    color = current_palette[0],
#)
#plt.plot(
#    x_range_original,
#    df.iloc[:, 2],
#    label = 'true High',
#    color = current_palette[2],
#)
#plt.plot(
#    x_range_original,
#    df.iloc[:, 3],
#    label = 'true Low',
#    color = current_palette[4],
#)
#plt.plot(
#    x_range_original,
#    df.iloc[:, 4],
#    label = 'true Close',
#    color = current_palette[6],
#)
#plt.plot(
#    x_range_original,
#    df.iloc[:, 5],
#    label = 'true Adj Close',
#    color = current_palette[8],
#)
#plt.xticks(x_range_original[::60], df.iloc[:, 0].tolist()[::60])
#plt.legend()
#plt.title('true market')
#plt.subplot(1, 2, 2)
#plt.plot(
#    x_range_future,
#    anchor(df_log[:, 0], 0.5),
#    label = 'predict Open',
#    color = current_palette[1],
#)
#plt.plot(
#    x_range_future,
#    anchor(df_log[:, 1], 0.5),
#    label = 'predict High',
#    color = current_palette[3],
#)
#plt.plot(
#    x_range_future,
#    anchor(df_log[:, 2], 0.5),
#    label = 'predict Low',
#    color = current_palette[5],
#)
#plt.plot(
#    x_range_future,
#    anchor(df_log[:, 3], 0.5),
#    label = 'predict Close',
#    color = current_palette[7],
#)
#plt.plot(
#    x_range_future,
#    anchor(df_log[:, 4], 0.5),
#    label = 'predict Adj Close',
#    color = current_palette[9],
#)
#plt.xticks(x_range_future[::60], date_ori[::60])
#plt.legend()
#plt.title('predict market')
#plt.show()



""""
plot the predicted turnover volume:
fig = plt.figure(figsize = (15, 10))
ax = plt.subplot(111)
ax.plot(x_range_original, df.iloc[:, -1], label = 'true Volume')
ax.plot(x_range_future, anchor(df_log[:, -1], 0.5), label = 'predict Volume')
box = ax.get_position()
ax.set_position(
    [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
)
ax.legend(
    loc = 'upper center',
    bbox_to_anchor = (0.5, -0.05),
    fancybox = True,
    shadow = True,
    ncol = 5,
)
plt.xticks(x_range_future[::30], date_ori[::30])
plt.title('overlap market volume')
plt.show()


fig = plt.figure(figsize = (20, 8))
plt.subplot(1, 2, 1)
plt.plot(x_range_original, df.iloc[:, -1], label = 'true Volume')
plt.xticks(x_range_original[::60], df.iloc[:, 0].tolist()[::60])
plt.legend()
plt.title('true market volume')
plt.subplot(1, 2, 2)
plt.plot(x_range_future, anchor(df_log[:, -1], 0.5), label = 'predict Volume')
plt.xticks(x_range_future[::60], date_ori[::60])
plt.legend()
plt.title('predict market volume')
plt.show()
"""