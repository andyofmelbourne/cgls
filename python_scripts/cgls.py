import numpy as np

class Steepest(object):
    """Run the steepest descent algorithm in general given the functions Adot and ATdot and the bvector.
    
    Solves A . x = b 
    given routines for A . x' and AT . b'
    and the bvector
    where x and b may be any numpy arrays."""

    def __init__(self, Adot, bvect, imax = 10**5, e_tol = 1.0e-10, x0 = None):
        self.Adot  = Adot
        self.bvect = bvect
        self.iters = 0
        self.imax  = imax
        self.e_tol = e_tol
        self.e_res = []
        if x0 == None :
            self.x = Adot(bvect)
            self.x.fill(0.0)
        else :
            self.x = x0

    def sd(self, iterations = None):
        """Iteratively solve the linear equations using the steepest descent algorithm.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iters == 0 :
            self.r   = self.bvect - self.Adot(self.x)
            self.d   = np.sum(self.r**2)
            self.d_0 = self.d.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            q     = self.Adot(self.r)
            alpha = self.d / np.sum(self.r * q)
            self.x = self.x + alpha * self.r
            if self.iters % 50 == 0 :
                self.r = self.bvect - self.Adot(self.x)
            else :
                self.r = r - alpha * q
            self.d     = np.sum(self.r**2)
            self.iters = self.iters + 1
            self.e_res.append(np.sqrt(self.d))
            if self.iters > self.imax or (self.d < self.e_tol**2 * self.d_0):
                break
        #
        return self.x


class Cgls(object):
    """Run the cgls algorithm in general given the functions Adot and ATdot and the bvector.
    
    Solves A . x = b 
    given routines for A . x' and AT . b'
    and the bvector
    where x and b may be any numpy arrays."""

    def __init__(self, Adot, bvect, imax = 10**5, e_tol = 1.0e-10, x0 = None):
        self.Adot  = Adot
        self.bvect = bvect
        self.iters = 0
        self.imax  = imax
        self.e_tol = e_tol
        self.e_res = []
        if x0 == None :
            self.x = Adot(bvect)
            self.x.fill(0.0)
        else :
            self.x = x0

    def cgls(self, iterations = None):
        """Iteratively solve the linear equations using the steepest descent algorithm.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iters == 0 :
            self.r     = self.bvect - self.Adot(self.x)
            self.d     = self.r.copy()
            self.d_new = np.sum(self.r**2)
            self.d_0   = self.d_new.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            q     = self.Adot(self.d)
            alpha = self.d_new / np.sum(self.d * q)
            self.x = self.x + alpha * self.d
            #
            if self.iters % 50 == 0 :
                self.r = self.bvect - self.Adot(self.x)
            else :
                self.r = r - alpha * q
            #
            d_old      = self.d_new.copy()
            self.d_new = np.sum(self.r**2)
            beta       = self.d_new / d_old
            self.d     = self.r + beta * self.d
            #
            self.iters = self.iters + 1
            self.e_res.append(np.sqrt(self.d))
            if self.iters > self.imax or (self.d < self.e_tol**2 * self.d_0):
                break
        #
        return self.x
