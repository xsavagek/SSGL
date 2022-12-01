# -*- coding: gbk -*-
#coding: unicode_escape

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tensorflow.contrib.layers as ly
import math
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
# 定义LSTM的超参数
# 根据股票历史数据中的最低价、最高价、开盘价、收盘价、交易量、交易额、跌涨幅等因素，对下一日股票最高价进行预测。
rnn_unit=6      # 隐含层数目
rnn_unit_d=1
input_size=6
input_size_d=1
output_size=6
output_size_d=1
lr1=0.00001
lr2=0.08
training_beta1=0.6
time_step=5        # 学习率
lambda_1 = 0.6
lambda_2 =0.4

# 定义权重和偏置
weights={
         'in1':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out1':tf.Variable(tf.random_normal([rnn_unit,output_size])),
         
        }
print(weights)
biases={
        'in1':tf.Variable(tf.constant(0.01,shape=[rnn_unit,])),
        'out1':tf.Variable(tf.constant(0.01,shape=[output_size,])),
       
       }
print(biases)



def lrelu(x, leak=0.4, name = 'lrelu'):
      
    return tf.maximum(x, leak*x, name = name)



def get_train_data(batch_size=50,time_step=5,train_begin=0,train_end=2200):
    
    batch_index=[]
    data_train=data[train_begin:train_end]

    # 标准化
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)

    # 训练集
    train_x,train_y=[],[]

    # 此时normalized_train_data的shape是n*8
    for i in range(len(normalized_train_data)-time_step):       # i = 1~5785

       # 生成batch_index：0，batch_size*1，batch_size*2
       if i % batch_size==0:
           batch_index.append(i)

       x=normalized_train_data[i:i+time_step,:7]                # x:shape 15*7
       y=normalized_train_data[i,6,np.newaxis]      # y:shape 15*1

       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))  # batch_index 收尾

    # train_x :n*15*7
    # train_y :n*15*1
    return batch_index,train_x,train_y



def get_test_data(time_step=5,test_begin=2300,batch_size=50):

    data_test=data[test_begin:]                 # 截取测试数据
    #mean=np.mean(data_test,axis=0)              # 平均数
    #std=np.std(data_test,axis=0)                # 方差
    #normalized_test_data=(data_test-mean)/std   # 标准化
    meanx,stdx=[],[]
    batch_index,test_x,test_y,realprice=[],[],[],[]
    for i in range(len(data_test)-time_step): 
       if i % batch_size==0:
           batch_index.append(i)

       x=data_test[i:i+time_step,:7]               
       y=data_test[i,6]
       meanx.append(np.mean(x,axis=0))
       stdx.append(np.std(x,axis=0))
       x=(x-meanx[i])/stdx[i]

       test_x.append(x.tolist())
       test_y.append(y.tolist())
       realprice.append(y.tolist())
    batch_index.append((len(data_test)-time_step))  # batch_index 收尾
  
    return meanx,stdx,batch_index,test_x,test_y,realprice    

    


def discriminator(Z,Y,reuse = False):
    
    with tf.variable_scope("discriminator") as scope:    
        if reuse:
            scope.reuse_variables()	    
        batch_size=tf.shape(Z)[0]   # 这行语句用于获取X的第一维的具体数据 获取batch_size值
        #time_step=tf.shape(Z)[1]    # 这行语句用于获取X的第一维的具体数据 获取time_step值
        
        
        rollZ = tf.reshape(Z,[batch_size,-1])
        rollY = tf.reshape(Y,[batch_size,-1])
        input1 = tf.concat([rollZ,rollY],1)
        input1 = tf.reshape(input1,[batch_size,36])
        
#        input1=tf.reshape(X,[batch_size,5])
        h1 = ly.fully_connected(input1,72,activation_fn=lrelu)
        h2 = ly.fully_connected(h1,100,activation_fn=lrelu)
        h3 = ly.fully_connected(h2,10,activation_fn=lrelu)
        h4 = ly.fully_connected(h3,1,activation_fn=None)
        pro2 = tf.nn.sigmoid(h4)
        
        return pro2,h4
	

def generator(Z, reuse=False):
    
    with tf.variable_scope("generator",reuse=tf.AUTO_REUSE) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size=tf.shape(Z)[0]   # 这行语句用于获取X的第一维的具体数据 获取batch_size值
        time_step=tf.shape(Z)[1]    # 这行语句用于获取X的第一维的具体数据 获取time_step值
        w_in=weights['in1']
        b_in=biases['in1']
    
        input=tf.reshape(Z,[time_step,7])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入 系统的输入单元数为7，所以将输入数据shape为n行7列
        input_rnn=lrelu(tf.matmul(input,w_in)+b_in)    # 输入数据与输入权重相乘，得到输入数据对隐含层的影响
    
        # 将tensor转成3维，作为lstm cell的输入
        # 隐含层的cell接收的数据是3维的，即将n*10的数据shape为n*15*10的数据
        input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    
        # 设置lstm的cell，BasicLSTMCell的入参有(self, num_units, forget_bias=1.0,state_is_tuple=True, activation=None, reuse=None)
        # 此处只设置其隐含层数目为rnn_unit，即10，其他参数使用默认值
        cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
        init_state=cell.zero_state(batch_size,dtype=tf.float32)
    
        # output_rnn是记录LSTM每个输出节点的结果，final_states是最后一个cell的结果
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, time_major=False, dtype=tf.float32)
    
        # 将输出数据shape为n*10格式
        output=output_rnn[:,-1,:]
        
        w_out=weights['out1']
        b_out=biases['out1']
        output=tf.reshape(output,[-1,input_size])
        output=lrelu(tf.matmul(output,w_out)+b_out)
        
        # cell输出经过与输出权重矩阵相乘并加入偏置后，得到最终输出
    
        pred=tf.reshape(output,[-1,input_size])
    
        return pred

def train_model(batch_size=50,time_step=5,train_begin=0,train_end=2300):
    

    Z=tf.placeholder(tf.float32, shape=[None,time_step,7])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,1])
    
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    
    
    #with tf.variable_scope("generator",reuse=tf.AUTO_REUSE):
    with tf.variable_scope("generator"):
    
       gen = generator(Z)
#    try:        
#        module_file = tf.train.latest_checkpoint('.')
#        saver.restore(sess, module_file)
#    except: 
    with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):

        D_real, D_logit_real = discriminator(Z,Y)
        D_fake, D_logit_fake = discriminator(Z,gen,reuse=True)

    D_loss = -tf.reduce_mean(tf.log(D_real)+tf.log(1.-D_fake))

#D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real))) 
#对判别器对真实样本的判别结果计算误差(将结果与1比较)  
#D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
 #对判别器对虚假样本的判别结果计算误差(将结果与0比较)  
#D_loss = D_loss_real + D_loss_fake #判别器的误差  
    MSE_loss = tf.reduce_mean(tf.square(tf.reshape(gen,[-1,6])-tf.reshape(Y, [-1,6])))              
#g_loss = tf.reduce_mean(1.-tf.log(D_fake))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    G_loss = lambda_1*MSE_loss+lambda_2*g_loss
    #G_loss = g_loss
#dreal_loss_sum = tf.summary.scalar("dreal_loss", D_loss_real) #记录判别器判别真实样本的误差  
#dfake_loss_sum = tf.summary.scalar("dfake_loss", D_loss_fake) #记录判别器判别虚假样本的误差  
    d_loss_sum = tf.summary.scalar("d_loss", D_loss) #记录判别器的误差  
    g_loss_sum = tf.summary.scalar("g_loss", G_loss) #记录生成器的误差

    merge = tf.summary.merge_all() 



    d_vars = [var for var in tf.trainable_variables() if "discriminator" in var.name]
    g_vars = [var for var in tf.trainable_variables() if "generator" in var.name]

    D_solver = tf.train.AdamOptimizer(learning_rate=lr1).minimize(D_loss, var_list=d_vars) #判别器的训练器  
    G_solver = tf.train.AdamOptimizer(learning_rate=lr2).minimize(G_loss, var_list=g_vars) #生成器的训练器  


    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('snapshots/', graph=sess.graph)  #日志记录器  
    for i in range(501):#for i in range(1001):
        
            # 按批次进行训练，每一批次80条数据
        for step in range(len(batch_index)-1):
            
#            _, D_loss_curr, d_loss_sum_value = sess.run([D_solver, D_loss, d_loss_sum], feed_dict={Z:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
#            _, G_loss_curr, g_loss_sum_value = sess.run([G_solver, G_loss, g_loss_sum], feed_dict={Z:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})  
            _, D_loss_curr, d_loss_sum_value = sess.run([D_solver, D_loss, d_loss_sum], feed_dict={Z:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            _, G_loss_curr, g_loss_sum_value = sess.run([G_solver, G_loss, g_loss_sum], feed_dict={Z:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})       
        print(i,'Dloss:',D_loss_curr,'Gloss:',G_loss_curr)
        summary_writer.add_summary(d_loss_sum_value,i)
        summary_writer.add_summary(g_loss_sum_value,i)
        if (i!=0) & (i % 500==0):#if (i!=0) & (i % 1000==0):
                print("保存模型：",saver.save(sess,'./stock2.model',global_step=i),"D_real",D_real,'\n',"D_logitreal",D_logit_real)
      
def prediction(time_step=5):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])

    # 获取测试数据的“均值”、“方差”、feature和label
    # test_x 16*20（最后一个长度为9） test_y：309
    mean,std, bei, test_x,test_y,realprice=get_test_data(time_step)

    with tf.variable_scope("generator",reuse=True):
        pred = generator(X)

    saver=tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:

        #参数恢复
        module_file = tf.train.latest_checkpoint('./')
        saver.restore(sess, module_file)
        test_predict=[]

        for step in range(len(bei)-1):
            prob=sess.run(pred,feed_dict={X:test_x[bei[step]:bei[step+1]]})
            predict=prob
            closeprice=predict[:,-1]
            test_predict.extend(closeprice)  
            
        realprice=realprice
        meanx = np.vstack((l for l in mean))
        stdx = np.vstack((l for l in std))
        test_predict=(np.array(test_predict)*stdx[:,-1])+(meanx[:,-1])
        acc=np.average(np.abs(test_predict-realprice[:len(test_predict)])/realprice[:len(test_predict)])  #偏差
        
        
        #return test_predict,acc
        print("len(test_predict):",len(test_predict),'\n',"len(test_y):",len(test_y),'\n',"acc:",acc)
        #以折线图表示结果
        
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(realprice))), realprice,  color='r')
        plt.show()
        return realprice,test_predict


f=open('./dataset/PAICCclose.csv',encoding='gbk')
df=pd.read_csv(f)               # 读入股票数据
data=df.iloc[:,1:].values     # 取第3-10列   data实际数据大小[6109 * 8]
print(data.shape)


#train_model()
rl,tp=prediction()

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

arr_true_close = rl
arr_pre_close = tp
np.savetxt('results/gan-lstm/PAICC/arr_true_close.csv', arr_true_close, delimiter = ',')
np.savetxt('results/gan-lstm/PAICC/arr_pre_close.csv', arr_pre_close, delimiter = ',')


print("RMSE = %0.4f" % get_rmse(arr_true_close, arr_pre_close))
print("MAPE = %0.4f%%" % get_mape(arr_true_close, arr_pre_close))
print("MAE = %0.4f" % get_mae(arr_true_close, arr_pre_close))

indicator = np.array([get_rmse(arr_true_close, arr_pre_close), 
                      get_mape(arr_true_close, arr_pre_close), 
                      get_mae(arr_true_close, arr_pre_close)])
np.savetxt('results/gan-lstm/PAICC/indicator.csv', indicator, delimiter = ',')





