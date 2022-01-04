# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 04:14:07 2021

@author: maout
"""


# optimal transport multidimensional reweighting
from pyemd import emd_with_flow
import numpy as np

from scipy.spatial.distance import pdist, squareform

__all__ = ["reweight_optimal_transport_multidim"]

def reweight_optimal_transport_multidim(samples, weights):

    """
    Computes deterministic transport map for particle reweighting.
    Particle state is multidimensional.

    Parameters
    ------------
        samples: array-like,
            Samples from distribution M x dim , with dim>=2.
        weights: array-like,
            weights for each sample M.

    Returns
    --------
        T: array like,
            transport map.

    Reweighting particles according to ensemble transform particle filter
    (ETPF) algorithm proposed by `Reich 2013`.
    Instead of particle resampling, ETPF employes Optimal Transport to 
    compute a deterministic particle shift which minimises the
    expected distances between the particles before and after the transformation.
    :math: `CO = X^T \\cdot X`
    :math: `CO = diag(CO)*ones(1,M) -2*CO + ones(M,1)*diag(CO)'`
    :math: `[dist,T] = emd(ww,ones(M,1)/M,CO,-1,3)`
    :math: `T = T \\cdot M`

    """

    num_samples = samples.shape[0] ## this should be the number of points

    covar = squareform(pdist(samples, 'euclidean'))
    b = np.ones((num_samples, 1)) / num_samples  # uniform distribution on samples

    _, T = emd_with_flow(weights.reshape(-1, ), b.reshape(-1, ), covar, -1)

    T = np.array(T)*num_samples


    return T    #%%
