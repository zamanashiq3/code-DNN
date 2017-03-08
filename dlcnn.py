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

#normalizing data
X_train = X_train/255
Y_train = Y_train/32767


#setting batch_size and epoch 

batchSize = 13
tt_epoch = 5

from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout, Activation, Flatten,Reshape,Permute
from keras.layers import Convolution2D, MaxPooling2D

#time to measure the experiment.
tt = time.time()

#model building starts here
seq = Sequential()

#first conv layer

seq.add(Convolution2D(16,2,2,border_mode='valid',batch_input_shape= 		(batchSize,1,53,53),input_shape=(1,53,53)))
seq.add(Activation('relu'))
seq.add(MaxPooling2D(pool_size=(2,2)))
seq.add(Dropout(0.25))

#second conv layer

seq.add(Convolution2D(16,2,2))
seq.add(Activation('relu'))
seq.add(MaxPooling2D(pool_size=(2,2)))
seq.add(Dropout(0.25))

#3rd conv layer

seq.add(Convolution2D(32,2,2))
seq.add(Activation('relu'))
seq.add(MaxPooling2D(pool_size=(2,2)))
seq.add(Dropout(0.25))

#flattening and adding lc layer for lstm pre-processing 

seq.add(Flatten())

#reshapeing and permute to feed the data into a lstm network

seq.add(Reshape((10,80)))
seq.add(Permute((2,1)))

#1st lstm layer

seq.add(LSTM(64,return_sequences=True,stateful=True))
seq.add(Activation('sigmoid'))
seq.add(Dropout(0.25))

#2nd lstm layer

seq.add(LSTM(32,stateful=True))
seq.add(Activation('sigmoid'))
seq.add(Dropout(0.25))

#a pre lc layer

seq.add(Dense(128))
seq.add(Activation('relu'))
seq.add(Dropout(0.5))


#final lc layer

seq.add(Dense(4702))
seq.add(Activation('relu'))


seq.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy'])

#checkpoint import
from keras.callbacks import ModelCheckpoint
from os.path import isfile, join
#weight file name
weight_file = 'dlcnn_weights.h5'

#loading previous weight file for resuming training 
if isfile(weight_file):
	seq.load_weights(weight_file)

#weight-checkmark
checkpoint = ModelCheckpoint(weight_file, monitor='acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

print('total time: ',time.time()-tt)

seq.fit(X_train,Y_train,batch_size=batchSize, 		  nb_epoch=tt_epoch,callbacks=callbacks_list)

#generating prediction for testing
pred = seq.predict(X_train,batch_size=batchSize,verbose=1)

pred = pred*32767
print('pred shape',pred.shape)
print('pred dtype',pred.dtype)
np.save('pred-dlcnn.npy',pred)
