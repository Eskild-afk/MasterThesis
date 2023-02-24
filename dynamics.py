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
    def create_path(self, stepsize:float, duration:float, seed=None)->np.array:
        steps = int(duration/stepsize)
        path = [self.init]
        time = [0]
        np.random.seed(seed)
        for i in range(steps):
            path.append((self.oneStep(path[i], stepsize)))
            time.append((i+1)*stepsize)

        return np.array(time), np.array(path)

    def forward_rate(self, time, start, end, initRate=None):
        if initRate==None: 
            initRate = self.init

        duration = end-start
        if time>=end:
            return 0

        return (self.ZCB(start-time,initRate)/self.ZCB(end-time,initRate)-1)/duration


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
        return stepfrom+(self.mean+self.rev*stepfrom)*stepsize+self.vol*np.sqrt(stepsize)*np.random.normal(0,1)

    def ZCB(self, duration, initRate=None):
        if duration == 0:
            return 1

        if initRate==None: 
            initRate = self.init
        
        m = -1/self.rev*(np.exp(self.rev*duration)-1)*initRate
        -self.mean/(np.square(self.rev))*(np.exp(self.rev*duration)-self.rev*duration-1)

        v = np.sqrt(
            np.square(self.vol)/np.square(self.rev)*(
                duration
                -1/(2*self.rev)*(1-np.exp(2*self.rev*duration))
                -2/self.rev*(np.exp(self.rev*duration)-1)
            )
        )
        return np.exp(m+np.square(v)/2)
    
    def expectedRate(self, duration, initRate=None):
        if initRate==None: 
            initRate = self.init
        
        return initRate*np.exp(self.rev*duration)+self.mean/self.rev*(np.exp(self.rev*duration)-1)



        
    