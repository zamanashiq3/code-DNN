"""
Multiple stacked lstm implemeation on the lip movement data.

Akm Ashiquzzaman
13101002@uap-bd.edu
Fall 2016

"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)
#random seed fixing for reproducibility

#data load & preprocessing 
X_train = np.load('../data/videopart43.npy').astype('float32')
Y_train = np.load('../data/audiopart43.npy').astype('float32')

X_train = X_train.reshape(826,13,53*53).astype('float32')
Y_train = Y_train.reshape(826,13*4702).astype('float32')

X_train = X_train/255
Y_train = Y_train/32767


from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM
import time

print("Building Model.....")
model_time = time.time()

#model starts here
model = Sequential()

#1st lstm layer
model.add(LSTM(512,input_shape=(13,53*53),batch_input_shape= 		(14,13,2809),return_sequences=True,stateful=True))
model.add(Dropout(0.25))

#2nd lstm layer
model.add(LSTM(256,return_sequences=True,stateful=True))
model.add(Dropout(0.25))

#2nd lstm layer
model.add(LSTM(128,stateful=True))
model.add(Dropout(0.5))

#final dense layer
model.add(Dense(13*4702))
model.add(Activation('tanh'))


model.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy'])

#checkpoint import
from keras.callbacks import ModelCheckpoint
from os.path import isfile, join
#weight file name
weight_file = '../weights/lstm_weight-v2.h5'

#loading previous weight file for resuming training 
if isfile(weight_file):
	model.load_weights(weight_file)

#weight-checkmark
checkpoint = ModelCheckpoint(weight_file, monitor='acc',verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

print("model compile time: "+str(time.time()-model_time)+'s')

# fit the model
model.fit(X_train,Y_train, nb_epoch=10, batch_size=14,callbacks=callbacks_list)

pred = model.predict(X_train,batch_size=14,verbose=1)

pred = pred*reshape(826*13,4702)
print('pred shape',pred.shape)
print('pred dtype',pred.dtype)
np.save('../predictions/pred-lstm-v2.npy',pred)
