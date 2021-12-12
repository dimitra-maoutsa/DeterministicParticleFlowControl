# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 04:14:07 2021

@author: maout
"""


# optimal transport multidimensional reweighting

import numpy as np
from pyemd import emd_with_flow
from scipy.spatial.distance import pdist,squareform
def reweight_optimal_transport_multidim(x,w):
    
    """
    Computes deterministic transport map for particle reweighting.
    Inputs:
    ------
        x: Samples from distribution M x dim , with dim>=2
        w: weights for each sample M
    
    Reweighting particles according to ensemble transform erticle filter algorithm proposed by Reich 2013
    Employes OT to compute a resampling scheme which minimises the expected distances
    between the particles before and after the resampling
    CO = X'*X;
    CO = diag(CO)*ones(1,M) -2*CO + ones(M,1)*diag(CO)';
    
    [dist,T] = emd_hat_mex(ww,ones(M,1)/M,CO,-1,3);
    T = T*M;
    
    """
    X = x
    M = x.shape[0] ## this should be the number of points
    
    CO = squareform(pdist(X, 'euclidean'))
    b =  np.ones((M,1)) / M  # uniform distribution on samples    
    
    dist, T = emd_with_flow(w.reshape(-1,),b.reshape(-1,), CO,-1)
    
    T = np.array(T)*M    
    
    
    return T    #%%
