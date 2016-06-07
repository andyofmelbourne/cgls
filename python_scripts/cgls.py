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
                self.r = self.r - alpha * q
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

    def __init__(self, Adot, bvect, ATdot = None, imax = 10**5, e_tol = 1.0e-10, x0 = None):
        self.Adot   = Adot
        self.ATdot  = ATdot
        self.bvect  = bvect
        self.iters  = 0
        self.imax   = imax
        self.e_tol  = e_tol
        self.e_res  = []
        if x0 is None :
            if self.ATdot is None :
                self.x = Adot(bvect)
            else :
                self.x = ATdot(bvect)
            self.x.fill(0.0)
        else :
            self.x = x0

    def cgls_symmetric_poisitve_definite(self, iterations = None):
        """Iteratively solve the linear equations using the steepest descent algorithm.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iters == 0 :
            self.r         = self.bvect - self.Adot(self.x)
            self.d         = self.r.copy()
            self.rTr_new   = np.sum(self.r**2)
            self.rTr_0     = self.rTr_new.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            Ad     = self.Adot(self.d)
            alpha  = self.rTr_new / np.sum(self.d * Ad)
            self.x = self.x + alpha * self.d
            #
            if self.iters % 1000 == 0 :
                self.r = self.bvect - self.Adot(self.x)
            else :
                self.r = self.r - alpha * Ad
            #
            rTr_old      = self.rTr_new.copy()
            self.rTr_new = np.sum(self.r**2)
            beta           = self.rTr_new / rTr_old
            self.d         = self.r + beta * self.d
            #
            self.iters = self.iters + 1
            #self.e_res.append(np.sqrt(self.d))
            if self.iters > self.imax or (self.rTr_new < self.e_tol**2 * self.rTr_0):
                break
        #
        return self.x

    def cgls(self, iterations = None):
        """Iteratively solve the linear equations using the steepest descent algorithm.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again.
        
        AT . A . x = AT . b   solves || A . x - b ||_min(x)
        
        d_0 = r_0 = AT . b - AT . A . x_0
        
        for i: 0 --> iters or while ||r_i|| / ||r_0|| < e_tol :
            alpha_i  = ||r_i|| / || A . d ||
            x_i+1    = x_i + alpha_i d_i
            r_i+1    = r_i - alpha_i AT . A . d_i
            beta     = r_i - ||r_i+1|| / ||r_i||
            d_i+1    = r_i+1 + beta d_i 
        """
        if self.iters == 0 :
            self.r         = self.ATdot(self.bvect) - self.ATdot(self.Adot(self.x))
            self.d         = self.r.copy()
            self.rTr_new   = np.sum(self.r**2)
            self.rTr_0     = self.rTr_new.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            Ad     = self.Adot(self.d)
            alpha  = self.rTr_new / np.sum(Ad * Ad)
            self.x = self.x + alpha * self.d
            #
            if self.iters % 1000 == 0 :
                self.r = self.ATdot(self.bvect) - self.ATdot(self.Adot(self.x))
            else :
                self.r = self.r - alpha * self.ATdot(Ad)
            #
            rTr_old        = self.rTr_new.copy()
            self.rTr_new   = np.sum(self.r**2)
            beta           = self.rTr_new / rTr_old
            self.d         = self.r + beta * self.d
            #
            self.iters = self.iters + 1
            #self.e_res.append(np.sqrt(self.d))
            if self.iters > self.imax or (self.rTr_new < self.e_tol**2 * self.rTr_0):
                break
        #
        return self.x
    
    def cgls_rect_not_very_good(self, iterations = None):
        """Iteratively solve the linear equations using the steepest descent algorithm.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again.
        """
        if self.iters == 0 :
            x = self.x.copy()
            R = self.bvect - self.Adot(x)
            s = np.zeros_like(self.x)
            S = np.zeros_like(R)
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            g = self.ATdot(R) # gradient:           g = AT . R
            G = self.Adot(g)  # conjugate gradient: G = A . g = A . AT . R

            # one step of steepest descent
            #if i == 0 : 
            if True : 
                alpha = np.sum( R * G )
                beta  = 0.
            
            # search the alpha G + beta S plane
            #if i != 0 :  
            if False :  
                GTG = np.sum(G * G)
                STS = np.sum(S * S)
                GTS = np.sum(G * S)

                det = GTG * STS - GTS * GTS
                #print det, GTG, STS, GTS, GTS
                #det = np.clip(det, 1e-20, np.inf)

                GTR = np.sum(G * R)
                STR = np.sum(S * R)
                alpha = ( STS * GTR - GTS * STR) / det
                beta  = (-GTS * GTR + GTG * STR) / det

            s = alpha * g + beta * s
            S = alpha * G + beta * S
            x = x + s
            R = R - S

        return x

def test_Cgls():
    import scipy.sparse.linalg
    M = 100
    N = 4 * 100
    iters = 5000
    A = np.random.random((N, M))
    x = np.random.random((M,))
    
    ATA = np.dot(A.T, A)
    
    b = np.dot(A, x)
    ATb = np.dot(A.T, b)
    
    import time 
    d0 = time.time()
    
    xret, info = scipy.sparse.linalg.cg(ATA, ATb, tol=1.0e-20, maxiter=iters)
    
    d1 = time.time()
    print 'scipy:', d1-d0, 'residual:', np.sum( (np.dot(ATA, xret) - ATb)**2 ), 'fidelity error:', np.sum( (xret-x)**2)
    
    
    Adot = lambda z : np.dot(ATA, z)
    cgls = Cgls(Adot, ATb, imax = 10**5, e_tol = 1.0e-20, x0 = None)
    
    d0 = time.time()
    
    xret = cgls.cgls_symmetric_poisitve_definite(iters)
    
    d1 = time.time()
    print 'Cgls:', d1-d0, 'residual:', np.sum( (np.dot(ATA, cgls.x) - ATb)**2 ), 'fidelity error:', np.sum( (xret-x)**2)
    
    
    Adot  = lambda z : np.dot(A, z)
    ATdot = lambda z : np.dot(A.T, z)
    cgls = Cgls(Adot, b, ATdot, imax = 10**5, e_tol = 1.0e-20, x0 = None)
    
    d0 = time.time()
    
    xret = cgls.cgls(iters)
    
    d1 = time.time()
    print 'Cgls:', d1-d0, 'residual:', np.sum( (np.dot(ATA, xret) - ATb)**2 ), 'fidelity error:', np.sum( (xret-x)**2)


