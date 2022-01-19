# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 02:23:15 2022

@author: maout
"""


import torch

__all__ = ['set_device']

def set_device():
    """
    Helper function to set the device where the tensors will be stored to the 
    available device.

    Returns
    -------
    device : TYPE
        'cuda' or 'cpu'.

    """
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