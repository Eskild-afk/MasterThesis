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

    


class Vasicek(Dynamic):
    '''
    The dynamic is (mean+rev*r)dt + vol*dW
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

    def ZCB(self, duration, time=None, initRate=None):
        if duration == 0:
            return 1

        if initRate==None: 
            initRate = self.init
        
        B = 1/self.rev*(np.exp(self.rev*duration)-1)
        
        firstpart = (np.square(self.vol)*(4*np.exp(self.rev*duration)-np.exp(2*self.rev*duration)-2*self.rev*duration-3))/(4*np.power(self.rev,3))
        
        secondpart = self.mean*(np.exp(self.rev*duration)-1-self.rev*duration)/(np.square(self.rev))
        
        A = firstpart+secondpart

        return np.exp(-A-B*initRate)
    
    def expectedRate(self, duration, initRate=None):
        if initRate==None: 
            initRate = self.init
        
        return initRate*np.exp(self.rev*duration)+self.mean/self.rev*(np.exp(self.rev*duration)-1)
    
    def forward_rate(self, time, start, end, initRate=None):
        if initRate==None: 
            initRate = self.init

        duration = end-start
        if time>end:
            return 0

        return (self.ZCB(duration=start-time,initRate=initRate)/self.ZCB(duration=end-time,initRate=initRate)-1)/duration



class G2PP(Dynamic):
    '''
    Two factor rate model
    r(t) = x(t) + y(t) + ϕ(t),
    dx(t) = -ax(t)dt + σ(t)dW1(t), x(0) = 0, (2.15)
    dy(t) = -by(t)dt + η(t)dW2(t), y(0) = 0
    dW1(t)dW2(t) = ρdt,
    '''
    def __init__(
            self, 
            initial,    #r/r_0
            a,
            b,
            sigma,
            eta,
            rho
        )->None:
        
        super().__init__(initial) 
        self.a      = a
        self.b      = b
        self.sigma  = sigma
        self.eta    = eta
        self.rho    = rho
    

    def create_path(self, stepsize:float, duration:float, seed=None):
        x = [0]
        y = [0]
        steps = int(duration/stepsize)

        np.random.seed(seed)
        rate = [self.init]
        time = [0]
        for i in range(steps):
            time.append((i+1)*stepsize)
            W1 = np.random.normal(0,1)
            W2 = self.rho*W1+np.sqrt(1-np.square(self.rho))*np.random.normal(0,1)
            dx = -self.a*x[i]+self.sigma*np.sqrt(stepsize)*W1
            dy = -self.b*y[i]+self.eta*np.sqrt(stepsize)*W2
            x.append(x[i]+dx)
            y.append(y[i]+dy)
            rate.append(rate[i]+dx+dy)

        self.x = x
        self.y = y
        self.time = np.array(time)
        return np.array(time), np.array(rate)
    
    def ZCB(self, duration, time, initRate=None):
        index = int(np.where(self.time==time)[0])
        x     = self.x[index]
        y     = self.y[index]

        if duration == 0:
            return 1

        if initRate==None: 
            initRate = self.init

        B1  = (1-np.exp(-self.a*duration))/self.a
        B2  = (1-np.exp(-self.b*duration))/self.b
        B12 = (1-np.exp(-(self.a+self.b)*duration))/(self.a+self.b)
        
        M  = x*B1+y*B2
        
        #V is split into three
        Vs  = np.square(self.sigma)/np.square(self.a)*(duration-B1-self.a/2*np.square(B1))
        Vs += np.square(self.eta)  /np.square(self.b)*(duration-B2-self.b/2*np.square(B2))
        Vs += 2*self.sigma*self.eta*self.rho/self.a/self.b*(duration-B1-B2+B12)

        return np.exp(-initRate*duration-M+0.5*Vs)
        # return np.exp(-M+0.5*Vs)

    def expectedRate(self, duration, initRate=None):
        pass
    
    def forward_rate(self, time, start, end, initRate=None):
        if initRate==None: 
            initRate = self.init

        duration = end-start
        if time>end:
            return 0

        return (self.ZCB(duration=start-time,time=time, initRate=initRate)/self.ZCB(duration=end-time, time=time, initRate=initRate)-1)/duration