

#NSS Feature Extraction for Logistic Regression

import numpy as np
import math

def estimateggd(Icap):
	gam=np.linspace(0.2,10,9800,endpoint=False)
	r_gam=np.array([(math.gamma(1.0/g) * math.gamma(3.0/g))/math.gamma(2.0/g)**2 for g in gam])	
	sigma_sq=np.mean(r_gam**2)
	sigma=np.sqrt(sigma_sq)
	E= np.mean(np.abs(Icap))
	rho= sigma_sq/np.power(E,2)
	array_position = np.argmin(np.abs(rho - r_gam))
	gamparam= gam[array_position]  
	return gamparam,sigma

def estimateaggd(Icap):
	gam=np.linspace(0.2,10,9800,endpoint=False)
	r_gam=np.array([(math.gamma(2.0/g)**2)/(math.gamma(1.0/g) * math.gamma(3.0/g)) for g in gam])
	leftstd=np.sqrt(np.mean([I**2 for I in Icap.ravel() if I<0]))
	rightstd=np.sqrt(np.mean([I**2 for I in Icap.ravel() if I>0]))	
	gammahat= leftstd/rightstd
	rhat= (np.mean(np.abs(Icap)))**2/np.mean(Icap**2)
	rhatnorm= (rhat*(gammahat**3 +1)*(gammahat+1))/((gammahat**2 +1)**2)
	array_position = np.argmin((r_gam - rhatnorm)**2)
	alpha= gam[array_position]
	return alpha,leftstd,rightstd
