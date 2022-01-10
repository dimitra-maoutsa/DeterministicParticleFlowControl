# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 07:20:39 2022

@author: maout
"""


import numpy as np

from scipy.spatial.distance import cdist
import torch
#from score_function_estimators import my_cdist
from typing import Union
from torch.autograd import grad
#%% select available device

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("GPU is not enabled in this notebook. \n"
          "If you want to enable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `GPU` from the dropdown menu")
  else:
    print("GPU is enabled in this notebook. \n"
          "If you want to disable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `None` from the dropdown menu")

  return device

#%%
  
class RBF(object):
    """
    Class for implementing a Gaussian RBF kernel in pytorch.
    
    Attributes
    ----------
    length_scale : float or list or numpy.array
        Length scale of the kernel. Either  single float for a single
        lengthscale across all dimension or a vector of floats for different
        lengthscales across each dimension. Defauls is 1.0
    signal_variance : float, optional
            This not getting used yet. The default is 1.0.
    device : Union[bool,str], optional
        Selected device where the computations will be executed,i.e. cpu or gpu.
        The default is None, which executes calculations on the cpu.
    multil : Union[bool, None], optional
        Boolean indicator determining whether lengthscale is a vector or a 
        single value. The default is False.
    K_data : numpy.ndarray
        Storage for the evaluation of the kernel on the datapoints (X, Y)
        in order to be reused in the calculations of the gradient of the Kernel.
        
        
     Methods
    -------
    Kernel(X, Y):
        Computes the kernel for the inouts X, and Y. Stores and returns 
        the result at K_data. Input arrays are of dimensionality (N, D) and 
        (M, D) respectively. Resulting Kernel has (N, M) dimension. 
    gradient_X(X, Y):
        Computes the gadient of the kernel with respect to the first argument
        along all D dimensions.
    
    
    
    """
    
    
    
    
    def __init__(self, length_scale: Union[float, torch.tensor, np.ndarray]=1.0, signal_variance: float=1.0, device: Union[bool,str]=None, multil: Union[bool, None]=False) -> None:
        """
        Initialising function for RBF Gaussian kernels using pytorch.
        Creates an object with necessary parammeters.

        Parameters
        ----------
        length_scale : Union[float, torch.tensor, np.ndarray], optional
            Lenghtscale estimated from data. Can be either a single float,
            or a vector for floats for different lengthscales for each dimension.
            The default is 1.0.
        signal_variance : float, optional
            This not getting used yet. The default is 1.0.
        device : Union[bool,str], optional
            Selected device where the computations will be executed,i.e. cpu or gpu.
            The default is None, which executes calculations on the cpu.
        multil : Union[bool, None], optional
            Boolean indicator determining whether lengthscale is a vector or a 
            single value. The default is False.
            TO DO: Remove this option and just check whether length_scale input 
            is a vector or a single float.

        Returns
        -------
        Instance of the object.

        """ 

        # initialize parameters
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.length_scale = torch.tensor(length_scale, dtype=torch.float32, device=self.device,
            requires_grad=True)        
        self.signal_variance = torch.tensor(signal_variance, dtype=torch.float32, device=self.device,
            requires_grad=True)
        self.multil = torch.tensor(multil, dtype=torch.bool, device=self.device, requires_grad=False)
        if self.multil:
            ##expand dimensions of lengtscale vector to enable broadcasting
            self.length_scale = self.length_scale[None, None,  :]
        self.K_data = torch.tensor(0, dtype=torch.float32, device=self.device, requires_grad=False)

        

    def Kernel(self, X: np.ndarray, Y: Union[bool, np.ndarray]=None) -> np.ndarray:

        if not torch.is_tensor(X):
          # convert inputs to pytorch tensors if not already pytorched
          X = torch.tensor(X, dtype=torch.float32, device=self.device) 
          N, D = X.shape       
        if Y is None:
            Y = X
        elif not torch.is_tensor(Y):
            Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
        M, _ = Y.shape
        # Re-indexing
        X_i = X[:, None, :] # shape (N, D) -> (N, 1, D)
        Y_j = Y[None, :, :] # shape (M, D) -> (1, M, D)
        
        if not self.multil: ##if a single lengthscale is provided
            # |X_i - Y_j|^2 # (N, M, D)
            sqd     = torch.sum( (X_i - Y_j)**2, 2)         
            # Divide by length scale            
            sqd  = torch.div(sqd, self.length_scale**2)
            K    = torch.exp( -0.5* sqd )
        else:          
          sqd1     = torch.div( (X_i - Y_j)**2, self.length_scale**2) 
          sqd     = torch.sum( sqd1, 2)  
          K    = torch.exp( -0.5* sqd )

        K   = torch.mul(self.signal_variance, K) # Signal Variance
        self.K_data = K
        return K#.detach().to_numpy()

    def gradient_X(self, X: np.ndarray, Y: Union[bool, np.ndarray]=None) -> np.ndarray:
        N, D = X.shape    
        M,_ = Y.shape
        diffs = X[:,None]-Y           
                  
        redifs = torch.div(diffs, self.length_scale**2)
        redifs = torch.einsum( 'ijk,ij->ijk', redifs, self.K_data)
        
              
        return redifs

        

    def gradient_X2(self, X):

        return None

    def gradient_XX(self,X: np.ndarray, Y: Union[bool, np.ndarray]=None) -> np.ndarray:

        # Convert to tensor that requires Grad
        X = torch.tensor(self.length_scale, dtype=torch.float32, device=self.device,requires_grad=True)

        if Y is None:
            Y = X
        else:
            Y = torch.tensor(Y, dtype=torch.float32, device=self.device, requires_grad=True)
        # compute the gradient kernel w.r.t. to the two inputs
        J = grad(self.__call__(X, Y))

        return J

    def gradient_XX2(self, X, Y=None):

        return None

#%% numpy versions of kernels functions
        
def Knp(x,y,l,multil=False):
    if multil:   
        res = np.ones((x.shape[0],y.shape[0]))                
        for ii in range(len(l)):             
            tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
            ##puts into tempi the cdist result
            tempi = cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),metric='sqeuclidean')
            res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
            
        return res
    else:
        tempi = np.zeros((x.shape[0], y.shape[0] ))
        
        tempi = cdist(x, y,'sqeuclidean') #this sets into the array tempi the cdist result
        return np.exp(-0.5*tempi/(l*l))
    
    
    
def grdx_K_all(x,y,l,multil=False): #gradient with respect to the 1st argument - only which_dim
      N,dim = x.shape    
      M,_ = y.shape
      diffs = x[:,None]-y                         
      redifs = np.zeros((1*N,M,dim))
      for ii in range(dim):          
      
          if multil:
              redifs[:,:,ii] = np.multiply(diffs[:,:,ii],Knp(x,y,l,True))/(l[ii]*l[ii])   
          else:
              redifs[:,:,ii] = np.multiply(diffs[:,:,ii],Knp(x,y,l))/(l*l)            
      return redifs


#%%

dim = 2
N = 3
M = 4
X = torch.randn(N, dim)
Z = torch.randn(M, dim)
# common device agnostic way of writing code that can run on cpu OR gpu
# that we provide for you in each of the tutorials
DEVICE = set_device()
dtype = torch.float


#%% test kernel evaluation with single lengthscale

lengthsc = 2
# pytorched
K_instance = RBF(length_scale=lengthsc, multil=False) ##instance of kernel object - non-evaluated
Ktorch = K_instance.Kernel(X, Z).detach().numpy()
gradK_torch = K_instance.gradient_X(X, Z).detach().numpy()
# numpyed
K_numpy = Knp(X.detach().numpy(), Z.detach().numpy(),l=lengthsc, multil=False).astype(np.float32)
grad_K_numpy = grdx_K_all(X.detach().numpy(), Z.detach().numpy(), l=lengthsc, multil=False).astype(np.float32)


np.testing.assert_allclose(Ktorch, K_numpy, rtol=1e-06)
np.testing.assert_allclose(gradK_torch, grad_K_numpy, rtol=1e-06)


#%% test kernel evaluation with multiple lengthscales
lengthsc = np.array([1,2])
# pytorched
K_instance2 = RBF(length_scale=lengthsc, multil=True) ##instance of kernel object - non-evaluated
Ktorch = K_instance2.Kernel(X, Z).detach().numpy()
gradK_torch = K_instance2.gradient_X(X, Z).detach().numpy()
# numpyed
K_numpy = Knp(X.detach().numpy(), Z.detach().numpy(),l=lengthsc, multil=True).astype(np.float32)
grad_K_numpy = grdx_K_all(X.detach().numpy(), Z.detach().numpy(), l=lengthsc, multil=True).astype(np.float32)


np.testing.assert_allclose(Ktorch, K_numpy, rtol=1e-06)
np.testing.assert_allclose(gradK_torch, grad_K_numpy, rtol=1e-06)

