from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,TimeDistributed,LSTM,Bidirectional
from keras.layers import Convolution2D,Flatten,MaxPooling2D
import time

print("Building Model.....")
model_time = time.time()

#model starts here
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

model.add(Bidirectional(LSTM(64,stateful=True)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.20))

model.add((Dense(512)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))

model.add((Dense(13*4702)))
model.add(Activation('sigmoid'))

model.summary()

from keras.utils.visualize_util import plot
plot(model, to_file='time-dnn.png')
