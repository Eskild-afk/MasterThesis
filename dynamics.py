import numpy as np

class Dynamic:
    '''
    Parent class for dynamics
    '''
    # Initializing dynamic with standard params
    def __init__(
            self, 
            initial
        )->None:
        
        self.init = initial

    #All dynamics should be able be simulated with a path
    def create_path(self, stepsize:float, duration:float, seed=None):
        steps = int(duration/stepsize)
        path = [self.init]
        np.random.seed(seed)
        for i in range(steps):
            path.append(self.oneStep(path[i], stepsize))
        return path


class Vasicek(Dynamic):
    '''
    Simple single factor Vasicek model with constant volatility
    initial is the starting rate
    mean is the mean reverting parameter
    reversion is how quickly it returns to the mean
    volatility doesn't need an explanation
    '''
    def __init__(
            self, 
            initial,    #r/r_0
            mean,       #theta
            reversion,  #kappa
            volatility  #sigma
        )->None:
        
        super().__init__(initial) 
        self.mean   = mean
        self.rev    = reversion
        self.vol    = volatility

    def oneStep(self, stepfrom, stepsize):
        return stepfrom+(self.mean-self.rev*stepfrom)*stepsize+self.vol*np.sqrt(stepsize)*np.random.normal(0,1)

        
    