"""
cnn+lstm implimentation on the lip movement data.

Akm Ashiquzzaman
13101002@uap-bd.edu
Fall 2016

after 1 epoch , val_acc: 0.0926

"""
from __future__ import print_function, division

#random seed fixing for reproducibility
import numpy as np
np.random.seed(1337)

import time

#Data loading 
X_train = np.load('videopart43.npy')
Y_train = np.load('audiopart43.npy')

#Reshaping to the 'th' order to feed into the cnn
X_train = X_train.reshape((X_train.shape[0],1,53,53)).astype('float32')
Y_train = Y_train.reshape((Y_train.shape[0],4702)).astype('float32')

#setting batch_size and epoch 

batchSize = 20
tt_epoch = 1

from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout, Activation, Flatten,Reshape,Permute
from keras.layers import Convolution2D, MaxPooling2D

#time to measure the experiment.
tt = time.time()

#model building starts here
seq = Sequential()

#first conv layer

seq.add(Convolution2D(256,5,5,border_mode='same',
				input_shape=(1,53,53)))
seq.add(Activation('relu'))
seq.add(MaxPooling2D(pool_size=(5,5)))
seq.add(Dropout(0.25))

#second conv layer

seq.add(Convolution2D(128,3,3))
seq.add(Activation('relu'))
seq.add(MaxPooling2D(pool_size=(3,3)))
seq.add(Dropout(0.25))


#flattening and adding lc layer for lstm pre-processing 

seq.add(Flatten())

#locally connected layers

seq.add(Dense(512))
seq.add(Activation('relu'))
seq.add(Dropout(0.5))

seq.add(Dense(256))
seq.add(Activation('relu'))
seq.add(Dropout(0.5))
#reshapeing and permute to feed the data into a lstm network

seq.add(Reshape((32,8)))
seq.add(Permute((2,1)))

#1st lstm layer

seq.add(LSTM(256,return_sequences=True))
seq.add(Activation('tanh'))
seq.add(Dropout(0.25))

#2nd lstm layer

seq.add(LSTM(128,return_sequences=True))
seq.add(Activation('tanh'))
seq.add(Dropout(0.25))

#3rd lstm layer

seq.add(LSTM(64))
seq.add(Activation('tanh'))
seq.add(Dropout(0.25))

#final lc layer

seq.add(Dense(4702))
seq.add(Activation('softmax'))


seq.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

#checkpoint import
from keras.callbacks import ModelCheckpoint
from os.path import isfile, join
#weight file name
weight_file = 'cnn_lstm_weight.h5'

#loading previous weight file for resuming training 
if isfile(weight_file):
	seq.load_weights(weight_file)

#weight-checkmark
checkpoint = ModelCheckpoint(weight_file, monitor='acc', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

print('total time: ',time.time()-tt)

seq.fit(X_train,Y_train,batch_size=batchSize, nb_epoch=tt_epoch
	,validation_split=0.2,callbacks=callbacks_list)

#generating prediction for testing
pred = seq.predict(X_train,batch_size=batchSize,verbose=1)

print('pred shape',pred.shape)
print('pred dtype',pred.dtype)
np.save('pred.npy',pred)
