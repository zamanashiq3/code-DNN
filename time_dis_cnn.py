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
X_train = np.load('videopart43.npy').astype('float32')
Y_train = np.load('audiopart43.npy').astype('float32')

#normalizing data
X_train = X_train/255
Y_train = Y_train/32767

X_train = X_train.reshape((826,13,1,53,53)).astype('float32')
Y_train = Y_train.reshape((826,13*4702)).astype('float32')


from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,TimeDistributed,LSTM,Bidirectional
from keras.layers import Convolution2D,Flatten,MaxPooling2D
import time

print("Building Model.....")
model_time = time.time()

model = Sequential()

model.add(TimeDistributed(Convolution2D(64, 3, 3,border_mode='valid'),batch_input_shape=(14,13,1,53,53),input_shape=(13,1,53,53)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))

model.add(TimeDistributed(Convolution2D(32, 2, 2, border_mode='valid')))
model.add(Activation('sigmoid'))


model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(256,return_sequences=True,stateful=True)))
model.add(Activation('sigmoid'))

model.add(Bidirectional(LSTM(128,return_sequences=True,stateful=True)))
model.add(Activation('sigmoid'))

model.add((LSTM(64,stateful=True)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.20))

model.add((Dense(512)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))

model.add((Dense(13*4702)))
model.add(Activation('sigmoid'))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

#checkpoint import
from keras.callbacks import ModelCheckpoint
from os.path import isfile, join
#weight file name
weight_file = 'time-cnn_weight.h5'

#loading previous weight file for resuming training 
if isfile(weight_file):
	model.load_weights(weight_file)

#weight-checkmark
checkpoint = ModelCheckpoint(weight_file, monitor='acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

print("model compile time: "+str(time.time()-model_time)+'s')

# fit the model
model.fit(X_train,Y_train, nb_epoch=1, batch_size=14,callbacks=callbacks_list)

pred = model.predict(X_train,batch_size=14,verbose=1)

pred = pred*32767
pred = pred*reshape(826*13,4702)
print('pred shape',pred.shape)
print('pred dtype',pred.dtype)
np.save('pred-time-cnn.npy',pred)
