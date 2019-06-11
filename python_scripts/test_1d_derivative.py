#!/usr/bin/env python

"""
We have f(x) 
And a test gradient function y(x) = df / dx 
I would like to test the gradient function by evaluating f(x) 
on a fine grid then evaluate the gradient numerically.
"""

import numpy as np
from pylab import *

if __name__ == '__main__':
    f = lambda x : x**2

    df = lambda x : 2. * x

    df_numpy = lambda x, f : np.gradient(f, np.gradient(x))    

    x = np.linspace(-10., 10, 10000, endpoint=True)

    f_x        = f(x)
    df_x       = df(x)
    df_numpy_x = df_numpy(x, f_x)

    clf()
    plot(x, f_x)
    plot(x, df_x)
    plot(x, df_numpy_x)
    show()

    mse = np.sum( (df_x - df_numpy_x)**2 ) / float(df_x.size)
    print('mean squared error:', mse)
    
