# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:59:38 2021

@author: maout
"""

import numpy as np

from scipy.spatial.distance import cdist

from score_function_estimators import my_cdist

X = np.random.random(size=(100,2))
Y = np.random.normal(size=(100,2))

inbuilt = cdist(X,Y, 'sqeuclidean')
mine = np.zeros((X.shape[0],Y.shape[0]))
my_cdist(X,Y, mine,dist='sqeuclidean')

#np.testing.assert_array_equal(mine, inbuilt)
np.testing.assert_allclose(mine, inbuilt)


X = np.zeros((100,2))
Y = np.zeros((100,2))

inbuilt = cdist(X,Y, 'sqeuclidean')
mine = np.zeros((X.shape[0],Y.shape[0]))
my_cdist(X,Y, mine,dist='sqeuclidean')

#np.testing.assert_array_equal(mine, inbuilt)
np.testing.assert_allclose(mine, inbuilt)