from scipy.integrate import quad
import numpy as np
def integrand(t, a):
	return pow(t,a-1)*pow(np.e,-t)


	
