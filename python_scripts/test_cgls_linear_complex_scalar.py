from cgls import *
import os, sys, getopt, inspect
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

def f(z):
    """This is the quadratic metric to minimise.
    
    Here f(x) = || (1 + 2i) . z - (-4 + 7i) ||
              = 5 zz* + (18 + i) z + (18 - i) z* + 65
              = zH . A . z - bT . z - bH . z* + c
    And so:
    A = 5
    b = -18 - i
    c = 65
    """
    out = 5.0 * np.conj(z) * z + (18.0 + 1.0J) * z + (18.0 - 1.0J) * np.conj(z) + 65.0
    return np.real(out)

def f_grad(x):
    """This is the derivative of f(x). When out = 0 we have the solution.
    
    Here we have: f'(x) = 4 . x - 40
    So our linear equations are A . x = b
                                4 . x = 40
    """
    out = 4.0 * x - 40.0
    return out
