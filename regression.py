## This is the code for simple logistic Regression


import cv2
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.applications import imagenet_utils
import h5py
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Dense


################################
def regression(train,trainlabels,test, testlabels):
	#testlabels[:]=float(testlabels[:])/100
	#trainlabels[:]=float(trainlabels[:])/100
	model = Sequential()
	
	model.add(Dense(64, input_dim=train.shape[1], activation='relu'))
	model.add(Dense(32, input_dim=train.shape[1], activation='relu'))
	model.add(Dense(8, input_dim=train.shape[1], activation='relu'))
	model.add(Dense(1,activation='sigmoid'))
	print model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
	model.fit(train,trainlabels, nb_epoch=1000, validation_data=(test, testlabels))
	print model.evaluate(train, trainlabels)
	print model.evaluate(test, testlabels)
	#print model.predict(test),testlabels

################################
'''
model = own_model()
hist = model.fit(train, trainlabels, epochs=3)
pred = model.predict(test)
print model.evaluate(train, trainlabels)
print model.evaluate(test, testlabels)
print pred

'''
'''
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
history = model.fit(train, trainlabels,epochs=4) 
score = model.evaluate(test, testlabels, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])
'''
