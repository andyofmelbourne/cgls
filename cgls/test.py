import os, sys, getopt, inspect
import numpy as np

import cgls_nonlinear as cg

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from pylab import cm

"""
f(x)      = sum (x-x0)**4
df(x)     = 4 (x-x0)**3
d2f(x)    = 12 (x-x0)**2
fd(x, d)  = sum d df(x)
dfd(x, d) = sum d d2f(x)
"""

"""
xsol = np.random.random((10,20))
f   = lambda x : np.sum( (x-xsol)**4 )
df  = lambda x : 4 * (x-xsol)**3
fd  = lambda x, d : np.sum(d * 4 * (x-xsol)**3)
dfd = lambda x, d : np.sum(12 * d**2 * (x-xsol)**2)

x0 = np.random.random((10,20))
t = x0.copy()
print(-1, f(t))

cgls = cg.Cgls(x0, f, df, fd, dfd=None)
for i in range(10):
    cgls.cgls(iterations=200)
    print(i, f(cgls.x))
"""


"""
f(p)     = sum_ij (p_i+1,j - p_i,j - gx_i,j)**2 + (p_i,j+1 - p_i,j - gx_i,j)**2
df(p_kl) = sum_ij 2( d(i+1=k) d(j=l) - d(i=k)d(j=l))
"""
def grad_ss(p):
    return p[1:, :] - p[:-1, :]

def grad_fs(p):
    return p[:, 1:] - p[:, :-1]

i, j = np.indices((101, 201))

phi = np.exp( -(i-20)**2/100. - (j-110)**2/200.)
phi = j.copy()

dx  = grad_ss(phi)[:, :-1]
dy  = grad_fs(phi)[:-1, :]

# mask
mask = np.ones(dx.shape, dtype=np.bool)
mask[20:30, 100:105] = False

dx  = dx * mask + dx.max()/5. * np.random.random(dx.shape)
dy  = dy * mask + dy.max()/5. * np.random.random(dx.shape)


#f = lambda x : (np.sum((grad_ss(x)[:,:-1]-dx)**2) + np.sum((grad_fs(x)[:-1,:]-dy)**2))
def f(x):
    return np.sum( mask * (grad_ss(x)[:,:-1]-dx)**2 ) + np.sum( mask * (grad_fs(x)[:-1,:]-dy)**2 )

def df(x):
    out = np.zeros_like(x)
    #
    # outi,j       = xi,j       - xi-1,j      - dxi-1,j
    out[1:, :-1]  += (x[1:, :-1] - x[:-1, :-1] - dx[:, :])*mask
    #
    # outi,j       = xi+1,j - xi,j - dxi,j
    out[:-1, :-1] -= (x[1:, :-1] - x[:-1, :-1] - dx[:, :])*mask
    #
    # outi,j       = xi,j       - xi,j-1      - dxi,j-1
    out[:-1, 1:]  += (x[:-1, 1:] - x[:-1, :-1] - dy[:, :])*mask
    #
    # outi,j       = xi,j+1       - xi,j      - dxi,j
    out[:-1, :-1] -= (x[:-1, 1:] - x[:-1, :-1] - dy[:, :])*mask
    return 2*out 

#assert(np.allclose(df(phi), 0*phi))

fd = lambda x, d : np.sum( d * df(x))

x0 = np.random.random(phi.shape)
xs = [x0.copy()]
cgls = cg.Cgls(x0, f, df, fd, dfd=None)
for i in range(1000):
    cgls.cgls(iterations=1)
    print(i, f(cgls.x))
    if i % 10 == 0 :
        xs.append(cgls.x.copy())
