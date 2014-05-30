import numpy as np

class Cgls(object):
    """Do an ILRUFT calculation in general given the functions Adot and ATdot and the bvector.
    
    Solves A . x = b 
    given routines for A . x' and AT . b'
    and the bvector
    where x and b may be any numpy arrays."""

    def __init__(self, Adot, ATdot, bvect):
        self.Adot  = Adot
        self.ATdot = ATdot
        self.bvect = bvect
        self.iterationsILRUFT = 0
        self.error_residual = []

    def ilruft(self, iterations = 10, refresh = 'no'):
        """Iteratively solve the linear equations using the conjugate gradient method.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if refresh == 'yes' :
            self.d     = np.copy(self.bvect)
            self.r     = self.ATdot(self.bvect)
            self.p     = np.copy(self.r)
            self.t     = self.Adot(self.p)
            self.norm_residual  = np.sum(np.abs(self.bvect)**2)

        if self.iterationsILRUFT == 0 :
            self.Sample = self.ATdot(self.bvect)
            self.Sample.fill(0)
            self.d     = np.copy(self.bvect)
            self.r     = self.ATdot(self.bvect)
            self.p     = np.copy(self.r)
            self.t     = self.Adot(self.p)
            self.norm_residual  = np.sum(np.abs(self.bvect)**2)
            self.error_residual = []
        
        for i in range(iterations):
            temp        = np.sum(np.abs(self.r)**2)
            self.alpha  = temp / np.sum(np.abs(self.t)**2)
            self.Sample += self.alpha * self.p
            self.d     -= self.alpha * self.t
            self.r      = self.ATdot(self.d)
            self.betta  = np.sum(np.abs(self.r)**2) / temp
            self.p      = self.r + self.betta * self.p
            self.t      = self.Adot(self.p)
            self.error_residual.append(np.sum(np.abs(self.d)**2)/self.norm_residual)
            #print 'residual error =', self.error_residual[-1]
            if self.error_residual[-1] <= 1.0e-30 :
                break

        self.iterationsILRUFT += iterations
        return self.Sample



class Steepest(object):
    """Run the steepest descent algorithm in general given the functions Adot and ATdot and the bvector.
    
    Solves A . x = b 
    given routines for A . x' and AT . b'
    and the bvector
    where x and b may be any numpy arrays."""

    def __init__(self, Adot, ATdot, bvect, imax = 10**5, e_tol = 1.0e-10, x0 = None):
        self.Adot  = Adot
        self.ATdot = ATdot
        self.bvect = bvect
        self.iters = 0
        self.error_residual = []
        self.imax = imax
        self.e_tol = e_tol
        self.x0 = x0

    def sd(self, iterations = None):
        """Iteratively solve the linear equations using the steepest descent algorithm.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iters == 0 :
            if self.x0 == None :
                self.x = self.ATdot(self.bvect)
                self.x.fill(0)
            else :
                self.x = self.x0
            self.iters = 0
            self.r = self.bvect - self.Adot(self.x)
            self.delta   = np.sum(self.r**2)
            self.delta_0 = self.delta
        # 
        if iterations == None:
            iterations = self.imax
        #
        for i in range(iterations):
            q     = self.Adot(self.r)
            alpha = self.delta / np.sum(self.r * q)
            self.x = self.x + alpha * self.r
            if self.iters % 50 == 0 :
                self.r = self.bvect - self.Adot(self.x)
            else :
                self.r = r - alpha * q
            self.delta   = np.sum(self.r**2)
            self.iters = self.iters + 1
            if self.iters > self.imax or (self.delta < self.e_tol**2 * self.delta_0):
                break
        #
        return self.x
