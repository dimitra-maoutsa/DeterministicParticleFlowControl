# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:14:34 2021

@author: maout
"""
import numpy as np

from scipy.spatial.distance import cdist

from score_function_estimators import my_cdist


def K(x,y,l,multil=False):
    if multil:                         
        res = np.ones((x.shape[0],y.shape[0]))  
        #print(len(l))              
        for ii in range(len(l)): 
            #print(ii)
            tempi = np.zeros((x.shape[0],y.shape[0]))
            ##puts into tempi the cdist result
            #my_cdist(x[:,ii:ii+1], y[:,ii:ii+1],tempi,'sqeuclidean')
            tempi = cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')            
            res = np.multiply(res, np.exp(-tempi/(2*l[ii]*l[ii])))                              
        return res
    else:
        tempi = np.zeros((x.shape[0], y.shape[0] ))                
        my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
        return np.exp(-tempi/(2*l*l))

def K1(x,y,l,multil=False):
    if multil:                
        res = np.ones((x.shape[0],y.shape[0]))                
        for ii in range(len(l)):             
            res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])) )
            
        return res
    else:
        return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
    
    
    
    
    
X = np.random.random(size=(4,2))
Y = np.random.normal(size=(4,2))

Ka = K(X,Y,1,False)
Kb = K1(X,Y,1,False)
#np.testing.assert_array_equal(mine, inbuilt)
np.testing.assert_allclose(Ka, Kb)



Kc = K(X,Y,np.ones(2),True)

Kd = K1(X,Y,np.ones(2),True)
#np.testing.assert_array_equal(mine, inbuilt)
np.testing.assert_allclose(Kc, Kd)

ls = np.ones(2)
ls[0] = 2
Ke = K(X,Y,ls,True)
Kf = K1(X,Y,ls,True)
#np.testing.assert_array_equal(mine, inbuilt)
np.testing.assert_allclose(Ke, Kf)


