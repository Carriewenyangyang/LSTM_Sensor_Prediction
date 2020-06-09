#!/usr/bin/env python
# coding: utf-8

# In[12]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv("C:/Users/lenovo/Desktop/LSTM_learn-master/Sensor-value1.csv") #*here
df.head()
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices =  range(i-history_size, i)
        data.append(np.reshape(dataset[indices],(history_size,1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


TRAIN_SPLIT = 80000     #*here
tf.random.set_seed(13)
#forecast a univariate time series
uni_data = df['C01424REG403RW']
uni_data.index = df['Timestamp']

#uni_data = df['T (degC)']      #*here
#uni_data.index = df['Date Time']   #*here
uni_data.head()
#uni_data.plot(subplots=True)
uni_data = uni_data.values

#standardize the data
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data - uni_train_mean)/uni_train_std


univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                          univariate_past_history,
                                          univariate_future_target)

x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                      univariate_past_history,
                                      univariate_future_target)

print("train_tensor[1]",x_val_uni[1])
print("label[1]",y_val_uni[0])

#simple example to predict the value at the red cross.

def create_time_steps(length):
    return list(range(-length,0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future','Model Prediction']
    marker = ['.-','rx','go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    #print("time_steps",time_steps)
    #print("plot_data",plot_data)
    
    if delta:
        future = delta
    else:
        future = 0
    
    plt.title(title)
    for i,x in enumerate(plot_data):
        
        if i:
            plt.plot(future, plot_data[i], marker[i],
                    markersize=10, label=labels[i])#for predicted values
        else:
            plt.plot(time_steps, plot_data[i].flatten(),
                    marker[i], label=labels[i])#for training dataset
    plt.legend() #for displaying labels
    plt.xlim([time_steps[0],(future+5)*2])#for the left number and right number
    
    plt.xlabel('Time-Step')
    plt.ylabel('x_train_uni[0]')
    return plt


#baseline
def baseline(history):
    return np.mean(history)


BATCH_SIZE = 128

BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

#input shape of data
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),#input_shape=(20,1) 
    tf.keras.layers.Dense(1) 
]) 


simple_lstm_model.compile(optimizer='adam', loss='mae')

#make a sample prediction
for x,y in val_univariate.take(1):
    print("simple_lstm_model.predict(x).shape",simple_lstm_model.predict(x).shape)
    print("val_univariate.take(1)",val_univariate.take(1))
    #print("x[0].numpy():y[0].numpy()",x.numpy(),y.numpy())

    
    
#train the model,save time for large dataset,epoch=200steps
EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                     steps_per_epoch = EVALUATION_INTERVAL,
                     validation_data = val_univariate, validation_steps=50)

#make a few predictions
for x,y in val_univariate.take(1):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                     simple_lstm_model.predict(x)[0]],
                    0, 'Simple LSTM model')
    

print("val_univariate.take(1)",val_univariate.take(1))
print("val_univariate.take(3)",val_univariate.take(3))

    
plot.show()

