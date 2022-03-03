# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 01:52:59 2022

@author: maout
"""


from typing import Union
import numpy as np
import torch 

__all__ = ['RBF']


class RBF:
    """
    RBF kernel class for pytorch implementation.
    
    
    """
    
    
    
    
    def __init__(self, length_scale: Union[float, torch.tensor, np.ndarray]=1.0, signal_variance: float=1.0, device: Union[bool,str]=None, multil: Union[bool, None]=False) -> None:
        """
        Kernel initialising function

        Parameters
        ----------
        length_scale : Union[float, torch.tensor, np.ndarray], optional
            Lengthscale of kernel. The default is 1.0.
        signal_variance : float, optional
            Variance of kernel. The default is 1.0.
        device : Union[bool,str], optional
            Device is either 'cpu' or 'cuda'/'cpu'. The default is None.
        multil : Union[bool, None], optional
            Boolean variable indicating whether the lengthscale is uniform across dimensions
            i.e. scalar, or a vector. True indicates a vector lengthscale.
            The default is False.

        Returns
        -------
        None
            Instance of the kernel.

        """
        # initialize parameters
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.length_scale = torch.tensor(length_scale, dtype=torch.float64, device=self.device,
            requires_grad=True)        
        self.signal_variance = torch.tensor(signal_variance, dtype=torch.float64, device=self.device,
            requires_grad=True)
        self.multil = torch.tensor(multil, dtype=torch.bool, device=self.device, requires_grad=False)
        if self.multil:
            ##expand dimensions of lengtscale vector to enable broadcasting
            self.length_scale = self.length_scale[None, None,  :]
        self.K_data = torch.tensor(0, dtype=torch.float64, device=self.device, requires_grad=False)

        

    def Kernel(self, X: np.ndarray, Y: Union[bool, np.ndarray]=None) -> torch.tensor:
        """
        Calculates the rbf gaussian kernel between data points X and Y.
        If Y is missing, computes the kernel between X with itself.
        

        Parameters
        ----------
        X : np.ndarray
            Data points, first entry -> (N x D).
        Y : Union[bool, np.ndarray], optional
            Data points, second entry -> (M x D). The default is None.

        Returns
        -------
        K:  torch.tensor
            Array of dimension NxM (or NxN if Y is missing) with the kernel evaluated
            at the data points.

        """

        if not torch.is_tensor(X):
          # convert inputs to pytorch tensors if not already pytorched
          X = torch.tensor(X, dtype=torch.float64, device=self.device) 
          #N, D = X.shape       
        if Y is None:
            Y = X
        elif not torch.is_tensor(Y):
            Y = torch.tensor(Y, dtype=torch.float64, device=self.device)
        M, _ = Y.shape
        # Re-indexing
        X_i = X[:, None, :] # shape (N, D) -> (N, 1, D)
        Y_j = Y[None, :, :] # shape (M, D) -> (1, M, D)
        
        if not self.multil: ##if a single lengthscale is provided
            
            sqd     = torch.sum( (X_i - Y_j)**2, 2)         # |X_i - Y_j|^2 # (N, M, D)
            # Divide by length scale   
            #print(sqd.device)
            #print(self.length_scale.device)        
            sqd  = torch.div(sqd, self.length_scale.to(self.device)**2)
            K    = torch.exp( -0.5* sqd )               
        else:          
          sqd1     = torch.div( (X_i - Y_j)**2, self.length_scale.to(self.device)**2) 
          sqd     = torch.sum( sqd1, 2)  
          K    = torch.exp( -0.5* sqd )

        K   = torch.mul(self.signal_variance, K) # Signal Variance
        self.K_data = K
        return K#.detach().to_numpy()

    def gradient_X(self, X: np.ndarray, Y: Union[bool, np.ndarray]=None) -> torch.tensor:
        """
        Computes the gradient of the kernel with respect to the first argument.

        Parameters
        ----------
        X : np.ndarray
            Data points, first entry -> (N x D).
        Y : Union[bool, np.ndarray], optional
            Data points, second entry -> (M x D). The default is None.

        Returns
        -------
        redifs : torch.tensor
            Array with the gradient of the Kernel -> (N, M, D).

        """
        #N, D = X.shape    
        M,_ = Y.shape
        diffs = X[:,None]-Y            
        #if self.multil:            
        redifs = torch.div(diffs, self.length_scale.to(self.device)**2)
        redifs = torch.einsum( 'ijk,ij->ijk', redifs, self.K_data)
        #redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
        #else:
            #redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)
              
        return redifs

        

    def gradient_X2(self, X):

        return None
    """
    def gradient_XX(self,X: np.ndarray, Y: Union[bool, np.ndarray]=None) -> torch.tensor:

        # Convert to tensor that requires Grad
        X = torch.tensor(length_scale, dtype=torch.float64, device=self.device,requires_grad=True)

        if Y is None:
            Y = X
        else:
            Y = torch.tensor(Y, dtype=torch.float64, device=self.device, requires_grad=True)
        # compute the gradient kernel w.r.t. to the two inputs
        J = grad(self.__call__(X, Y))

        return J
    """
    def gradient_XX2(self, X, Y=None):

        return None
