import numpy as np
from scipy import stats, optimize, integrate
from scipy.stats import norm, multivariate_normal
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
    def create_path(self, stepsize:float, duration:float, fwd=0, seed=None)->np.array:
        steps = int(duration/stepsize)
        path = [self.init]
        np.random.seed(seed)
        Z = np.random.normal(0,1,steps)
        for i in range(steps):
            path.append((self.oneStep(t=(i+1)*stepsize, stepfrom=path[i], stepsize=stepsize, fwd=fwd, Z=Z[i])))

        return np.arange(0,duration+stepsize,stepsize), np.array(path)


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