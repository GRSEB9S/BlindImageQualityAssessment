
### This Code Is for Training Convolutional Neural Network (CNN)
### Data set used: LIVEIQA (from Alan Bovik's Website)


from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image
import numpy as np
import cv2
import regression
from keras.models import model_from_json

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 30 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64
conv_depth_3 = 128
conv_depth_4 = 256
conv_depth_5 = 512 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons



def probabilisticvecs(yi,m):
	B=64
	den=0
	sum1=0
	pvrs=[]
	for i in range(m):
		den=den+np.exp(-B*pow((float(yi)/100)-(((i*10)+5.0)/100),2))
	for i in range(m):
		p1=np.exp(-B*pow((float(yi)/100)-(((i*10)+5.0)/100),2))
		sum1=sum1+p1/den
		pvrs.append(p1/den)
	return pvrs

def onehot(yi,m):
	oh=[0]*m
	yi=float(yi)
	for i in range(0,m):
		if i*10<=yi and (i+1)*10>yi:
			oh[i]=1
	return oh

folders=["fastfading","gblur","jp2k","jpeg","wn"]
i=0
for folder in folders:
	with open("./"+folder+"/info.txt", "r") as f:
		data = f.readlines()
	traini=np.array([])
	File=[]
	for line in data:
		words = line.split()
		File.append(words)
	

	traini = np.array([cv2.resize(cv2.imread("./"+folder+"/"+fname[1]),(100,100)) for fname in File ])
	#trainlabelsi= np.array([float(f[2]) for f in File])
	if i==0:
		train=traini
	#	trainlabels=trainlabelsi
		i=1	
	else :	
		train=np.append(train,traini,axis=0)
	#	trainlabels=np.append(trainlabels,trainlabelsi,axis=0)		
trainlabels=np.loadtxt("csvlist.dat",delimiter=",")

print trainlabels

i=0
for a in range(len(trainlabels)):
	if i==0:
		patch=image.extract_patches_2d(train[a], (64,64),0.005)
		patchlabel=np.array(patch.shape[0]*[trainlabels[a]])
		i=1
	else:
		patchi=image.extract_patches_2d(train[a], (64,64),0.005)
		patchlabeli=np.array(patchi.shape[0]*[trainlabels[a]])
		patch=np.append(patch,patchi,axis=0)
		patchlabel=np.append(patchlabel,patchlabeli,axis=0)		


print patch.shape
print patchlabel.shape

X_train, X_test, Y_traintarget, Y_testtarget = train_test_split(patch, patchlabel, test_size=0.2, random_state=0)
j=0
for Y in Y_traintarget:
	Y_traini= np.array([probabilisticvecs(Y,10)])
	if j==0:
		Y_train=Y_traini
		j=1
	else:
		Y_train=np.append(Y_train,Y_traini,axis=0)
j=0
for Y in Y_testtarget:
	Y_testi= np.array([probabilisticvecs(Y,10)])
	if j==0:
		Y_test=Y_testi
		j=1
	else:
		Y_test=np.append(Y_test,Y_testi,axis=0)

#print train.shape,trainlabels.shape

#print ">>>>>>>>>>>>>>>>>>>>>"

print X_train.shape,Y_train.shape

print ">>>>>>>>>>>>>>>>>>>>>"

print X_test.shape,Y_test.shape

num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10 
num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = 10 # there are 10 image classes

n=X_train.shape[0]
X_train1 = X_train[:, :n/2].astype('float32')
X_train2 = X_train[:, n/2:].astype('float32') 
X_train = np.hstack((X_train1, X_train2))

X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

#Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
#Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels
Y_train=Y_train
Y_test=Y_test

with open('X_train', 'wb') as fp:
	pickle.dump(X_train, fp)

with open('Y_train', 'wb') as fp:
	pickle.dump(Y_train, fp)

with open('X_test', 'wb') as fp:
	pickle.dump(X_test, fp)

with open('Y_test', 'wb') as fp:
	pickle.dump(Y_test, fp)
'''

###############################################################################################################
'''
inp = Input(shape=(height, width, depth))
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), activation='relu')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
conv_2 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), activation='relu')(conv_1)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)

conv_3 = Convolution2D(conv_depth_3, (kernel_size, kernel_size), activation='relu')(conv_2)
pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
conv_4 = Convolution2D(conv_depth_4, (kernel_size, kernel_size), activation='relu')(conv_3)
pool_4 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
conv_5 = Convolution2D(conv_depth_4, (kernel_size, kernel_size), activation='relu')(conv_4)
pool_5 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_5)

drop = Dropout(drop_prob_1)(pool_5)

flat = Flatten()(drop)
hidden = Dense(hidden_size, activation='relu')(flat)
out = Dense(num_classes, activation='softmax')(hidden)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

print model.evaluate(X_train, Y_train, verbose=1)
print model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!
#X_trainvecs=model.predict(X_train)
#X_testvecs=model.predict(X_test)


#regression.regression(X_trainvecs,Y_traintarget/100,X_testvecs,Y_testtarget/100)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
#######################################################################################################################

''' 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''

