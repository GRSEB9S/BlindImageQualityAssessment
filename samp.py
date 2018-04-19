## Feature Extraction


import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
import numpy as np
import func as f
from scipy.integrate import quad
import ggd
import math
from sklearn.model_selection import train_test_split
import pickle
#implementation of Icap
folders=["fastfading","gblur","jp2k","jpeg","wn"]
i=0
for folder in folders:
	with open("/home/santosh/Documents/BTP/databaserelease2/"+folder+"/info.txt", "r") as f:
		data = f.readlines()
	traini=np.array([])
	File=[]
	for line in data:
		words = line.split()
		File.append(words)
	

	traini = np.array([cv2.resize(cv2.imread("/home/santosh/Documents/BTP/databaserelease2/"+folder+"/"+fname[1]),(50,50)) for fname in File ])
	trainlabelsi= np.array([float(f[2]) for f in File])
	if i==0:
		train=traini
		trainlabels=trainlabelsi
		i=1	
	else :	
		train=np.append(train,traini,axis=0)
		trainlabels=np.append(trainlabels,trainlabelsi,axis=0)		


X_train, X_test, Y_traintarget, Y_testtarget = train_test_split(train, trainlabels, test_size=0.2, random_state=0)
j=0



window = signal.gaussian(49, std=7)
window=window/sum(window)
features=[]
for I in X_train:
	#I = cv2.imread('img5.bmp',cv2.IMREAD_GRAYSCALE)
	I = I.astype(np.float32)
	u = cv2.filter2D(I,-1,window)
	u = u.astype(np.float32)

	diff=pow(I-u,2)
	diff = diff.astype(np.float32)
	sigmasquare=cv2.filter2D(diff,-1,window)
	sigma=pow(sigmasquare,0.5)

	Icap=(I-u)/(sigma+1)
	Icap = Icap.astype(np.float32)
	gamparam,sigma = ggd.estimateggd(Icap)
	feat=[gamparam,sigma]



	shifts = [ (0,1), (1,0) , (1,1) , (-1,1)];
	for shift in shifts:
		shifted_Icap= np.roll(Icap,shift,axis=(0,1))
		pair=Icap*shifted_Icap
		alpha,leftstd,rightstd=ggd.estimateaggd(pair)
		const=(np.sqrt(math.gamma(1/alpha))/np.sqrt(math.gamma(3/alpha)))
		meanparam=(rightstd-leftstd)*(math.gamma(2/alpha)/math.gamma(1/alpha))*const;
		feat=feat+[alpha,leftstd,rightstd,meanparam]
		
	feat=np.array(feat)
	features.append(feat)
	print features

with open('features', 'wb') as fp:
	pickle.dump(features, fp)

#with open ('outfile', 'rb') as fp:
#	itemlist = pickle.load(fp)
#print alpha,leftstd,rightstd


#implemetaion of f(x,alpha,sigmasquare)

#alpha=0.09
#B=sigma * pow(quad(f.integrand,0,float('inf'),args=(1/alpha))[0]/quad(f.integrand,0,float('inf'),args=(3/alpha))[0],0.5)
#print B

#f1=alpha/(2*B*quad(f.integrand,0,float('inf'),args=(1/alpha))[0])

#f2=pow(np.e,-pow(abs(Icap)/B,alpha))
#f=f1*f2

#print f




'''
print Icap.ravel()
k=plt.hist(Icap.ravel(),360,[-3,3])
print sum(k[0])/len(k[0])
print len(k[1])
plt.title('H - Histogram')
plt.show()
plt.subplot(121),plt.imshow(I),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(Icap),plt.title('Icap')
plt.xticks([]), plt.yticks([])

plt.show()
'''
