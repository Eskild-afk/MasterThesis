import numpy as np
from scipy import stats, optimize
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
        return stepfrom+(self.mean-self.rev*stepfrom)*stepsize+self.vol*np.sqrt(stepsize)*np.random.normal(0,1)

    def B(self, duration):
        return (1-np.exp(-self.rev*(duration)))/self.rev

    def A(self, duration):
        sigma = self.vol
        b     = self.mean
        beta  = self.rev
        return (np.square(sigma)/2/np.square(beta)-b/beta)*(self.B(duration)-(duration))+np.square(sigma)/4/beta*np.square(self.B(duration))

    def ZCB(self, duration, time=None, initRate=None):
        if duration == 0:
            return 1

        if initRate==None: 
            initRate = self.init
        
        B = self.B(duration)
        
        A = self.A(duration)

        return np.exp(-A-B*initRate)
    
    def expectedRate(self, duration, initRate=None):
        if initRate==None: 
            initRate = self.init
        
        return initRate*np.exp(-self.rev*duration)+self.mean/self.rev*(1-np.exp(-self.rev*duration))
    
    def forward_rate(self, time, start, end, initRate=None):
        if initRate==None: 
            initRate = self.init

        duration = end-start
        if time>end:
            return 0

        return (self.ZCB(duration=start-time,initRate=initRate)/self.ZCB(duration=end-time,initRate=initRate)-1)/duration
    
    def M(self, s, t, fwd):
        sigma = self.vol
        b     = self.mean
        beta  = self.rev

        return (b-np.square(sigma)/beta)*self.B(fwd-t)+np.square(sigma)/beta*self.B(2*t-2*s)*np.exp(-beta*(fwd-t))

    def meanFwd(self, s,t, fwd):
        rs      = self.init
        beta    = self.rev

        return rs*np.exp(-beta*(t-s))+self.M(s,t,fwd)
    
    def varFwd(self, s,t):
        sigma   = self.vol
        return np.square(sigma)/2*self.B(2*t-2*s)

    def rbarhelper(self, t, rbar, fixedSchedule, floatingSchedule, fixedRate):
        sum = 0
        if len(floatingSchedule) == 0:
            TR=10
        else:
            TR=floatingSchedule[0]
        for i in fixedSchedule[1::]:
            c=fixedRate
            
            if i == fixedSchedule[-1]:
                c+=1    
        
            sum += c*self.ZCB(i-t, time=t, initRate=rbar)
        D = self.ZCB(TR-t, initRate = rbar)
        return D-sum

    def ZCBPut(self, time, T, S, Strike):
        sigma   = self.vol
        beta    = self.rev
        # Taken from Brigo (3.41)
        sigmap = sigma*np.sqrt((1-np.exp(-2*beta*(T-time)))/(2*beta))*self.B(S-T)
        h = np.log(self.ZCB(S-time)/self.ZCB(T-time)/Strike)/sigmap+sigmap/2
        
        return Strike*self.ZCB(T-time)*stats.norm.cdf(-h+sigmap)-self.ZCB(S-time)*stats.norm.cdf(-h)


    def PswaptionSC(self, time, expiry, fixedSchedule, floatingSchedule, fixedRate):
        
        floatingSchedule = floatingSchedule[floatingSchedule>expiry-0.5]
        if len(floatingSchedule) == 0:
            TR=10
        else:
            TR=floatingSchedule[0]

        fixedSchedule = fixedSchedule[fixedSchedule>expiry-1]
        
        #finding rBar
        rbar   = optimize.fsolve(func=lambda r: self.rbarhelper(t=expiry, rbar=r, fixedSchedule=fixedSchedule, floatingSchedule = floatingSchedule, fixedRate=fixedRate), x0=self.init)[0]
        sum = 0

        for Si in fixedSchedule[1::]:
            ci = fixedRate
            if Si == fixedSchedule[-1]:
                ci += 1

            Xi = np.exp(-self.A(Si-expiry)-self.B(Si-expiry)*rbar)

            if ci*self.ZCBPut(time, expiry, Si, Xi)==ci*self.ZCBPut(0, expiry, Si, Xi):
                sum += ci*self.ZCBPut(time, expiry, Si, Xi)

        return sum



        # #Mean and variance under expiry forward measure
        # fwdMean = self.meanFwd(0, TR, expiry)
        # fwdVar  = self.varFwd(0, TR)

        # #finding rBar
        # rbar    = optimize.fsolve(func=lambda r: self.rbarhelper(rbar=r, fixedSchedule=fixedSchedule, fixedRate=fixedRate), x0=self.init)[0]

        # #Prob that brackets are po
        # prob    = stats.norm.cdf((fwdMean-rbar)/fwdVar)

        # return prob*(self.ZCB(TR)+self.rbarhelper(self.init, fixedSchedule, fixedRate)-1)
    
    def RswaptionSC(self, expiry, fixedSchedule, floatingSchedule, fixedRate):
        floatingSchedule = floatingSchedule[floatingSchedule>=expiry]
        TR=floatingSchedule[0]

        fixedSchedule = fixedSchedule[fixedSchedule>=expiry]
        
        #Mean and variance under expiry forward measure
        fwdMean = self.meanFwd(0, TR, expiry)
        fwdVar  = self.varFwd(0, TR)

        #finding rBar
        rbar    = optimize.fsolve(func=lambda r: self.rbarhelper(rbar=r, fixedSchedule=fixedSchedule, fixedRate=fixedRate), x0=self.init)[0]

        prob    = stats.norm.cdf((rbar-fwdMean)/fwdVar)

        return prob*(self.rbarhelper(self.init, fixedSchedule, fixedRate)-1-self.ZCB(TR))




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