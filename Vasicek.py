from dynamics import Dynamic
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import fsolve, minimize
from scipy import integrate

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

    def oneStep(self, t, stepfrom, stepsize, fwd=0):
        return stepfrom+(self.mean-self.rev*stepfrom)*stepsize+self.vol*np.sqrt(stepsize)*np.random.normal(0,1)-np.square(self.vol)*self.B(fwd-0)*stepsize

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
    
    def expectedRate(self, duration, fwd=0, initRate=None):
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
    
    def meanR(self, duration, initRate=None):
        if initRate==None: 
            initRate = self.init

        beta    = self.rev
        b       = self.mean

        return initRate*np.exp(-beta*duration)+b/beta*(1-np.exp(-beta*duration))
    
    def variance(self, s,t):
        sigma   = self.vol
        beta    = self.rev

        return np.square(sigma)/2/beta*(1-np.exp(-2*beta*(t-s)))

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
        return self.variance(s,t)
    
    def cov(self,s,t):
        sigma = self.vol
        beta  = self.rev
        return np.square(sigma)*np.exp(-beta*(s+t))*(np.exp(2*beta*np.minimum(s,t))-1)/(2*beta)

    def corr(self,s,t):
        return self.cov(s,t)/np.sqrt(self.variance(0,s)*self.variance(0,t))

    def rbarhelper(self, t, expiry, rbar, fixedSchedule, floatingSchedule, fixedRate):
        sum = 0
        if len(floatingSchedule) == 0:
            TR=10
        else:
            TR=floatingSchedule[0]
        for i in fixedSchedule[1::]:
            c=fixedRate
            
            if i == fixedSchedule[-1]:
                c+=1    
        
            sum += c*self.ZCB(i-expiry, time=t, initRate=rbar)

        # D = self.ZCB(expiry-t, initRate = rbar)
        return 1-sum

    def ZCBPut(self, time, T, S, Strike):
        sigma   = self.vol
        beta    = self.rev
        # Taken from Brigo (3.41)
        sigmap = sigma*np.sqrt((1-np.exp(-2*beta*(T-time)))/(2*beta))*self.B(S-T)
        h = (np.log(self.ZCB(S-time))-np.log(self.ZCB(T-time)*Strike))/sigmap+sigmap/2
        if (abs(h)>=1000) or np.isnan(h):
            return 10000
        return Strike*self.ZCB(T-time)*norm.cdf(-h+sigmap)-self.ZCB(S-time)*norm.cdf(-h)

    def ZCBCall(self, time, T, S, Strike):
        sigma   = self.vol
        beta    = self.rev
        # Taken from Brigo (3.41)
        sigmap = sigma*np.sqrt((1-np.exp(-2*beta*(T-time)))/(2*beta))*self.B(S-T)
        h = (np.log(self.ZCB(S-time))-np.log(self.ZCB(T-time)*Strike))/sigmap+sigmap/2
        if (abs(h)>=1000) or np.isnan(h):
            return 10000
        return self.ZCB(S-time)*norm.cdf(h)-Strike*self.ZCB(T-time)*norm.cdf(h-sigmap)
    
    def swap(self, time, fixedSchedule, floatingSchedule, fixedRate, initRate=None):
        try:
            TR = floatingSchedule[0]
        except:
            TR = floatingSchedule
        if time > TR:
            print("Time is after first fixed payment")
            return None
        
        sum = 0
        for i in fixedSchedule[1::]:
            c=fixedRate
            if i == fixedSchedule[-1]:
                c+=1
            sum += c*self.ZCB(i-time, time=time, initRate=initRate)
        return self.ZCB(TR-time)-sum

    def swaptionGivenRates(self,expiryRate,firstResetRate,expiry,fixedSchedule,floatSchedule, fixedRate, payer=True):
        if payer:
            w=1
        else:
            w=-1 
        TR =   floatSchedule[floatSchedule>=expiry-0.5][0] #First reset date
        TRp1 = floatSchedule[floatSchedule>=expiry-0.5][1] #Second reset date
        #Fixed payment dates after first reset date (no fixed payment on the first reset)
        Si = fixedSchedule[fixedSchedule>TR]               
        cisum = 0
        for i in Si:
            c = fixedRate
            if i == Si[-1]:
                c+=1
            cisum += c*self.ZCB(i-expiry,initRate=expiryRate)
        
        if (TR==0) and  (expiry>TR):
            mean    = self.meanFwd(0,expiry,expiry)
            var     = self.varFwd(0,expiry)
            result  = self.ZCB(TRp1-expiry,initRate=expiryRate)/self.ZCB(TRp1-TR, initRate=self.init)-cisum
            prob    = norm.pdf(expiryRate,mean,np.sqrt(var))

        elif expiry   <=  TR:
            mean    = self.meanFwd(0,TR,expiry)
            var     = self.varFwd(0,TR)
            result  = 1/self.ZCB(TRp1-TR, initRate=expiryRate)-cisum
            prob    = norm.pdf(expiryRate,mean,np.sqrt(var))
        
        elif expiry>TR:
            xy=np.array([firstResetRate,expiryRate])
            meanM   = np.array([self.meanFwd(0,TR,expiry),self.meanFwd(0,TRp1,expiry)])
            covM    = np.array([[self.varFwd(0,TR), self.cov(TR,expiry)],[self.cov(expiry,TR),self.varFwd(0,expiry)]])
            result  = self.ZCB(TRp1-expiry,initRate=expiryRate)/self.ZCB(TRp1-TR, initRate=firstResetRate)-cisum
            prob    = multivariate_normal.pdf(xy, mean=meanM, cov=covM)

        return np.maximum(w*result,0)*prob
            
    def swaptionIntegration(self,expiry,fixedSchedule,floatSchedule, fixedRate, payer=True):
        '''
        This function requires that the first reset date is after the current time t
        and if t=TR then it requires that expiry>=TRp1
        '''
        
        TR = floatSchedule[floatSchedule>=expiry-0.5][0] #First reset date
        TRp1 = floatSchedule[floatSchedule>=expiry-0.5][1]
        if expiry<TR:
            
            pass
        elif (TR==0) and (expiry>TR):
            toIntegrate = lambda x: self.swaptionGivenRates(x,self.init,expiry,fixedSchedule,floatSchedule, fixedRate, payer)
            expirySd      = 5*np.sqrt(self.varFwd(0,expiry))
            return self.ZCB(expiry)*integrate.quad(toIntegrate, -5*expirySd, 5*expirySd)[0]
        elif expiry >   TR:     #Two rates case (double integral)
            toIntegrate = lambda x,y: self.swaptionGivenRates(x,y,expiry,fixedSchedule,floatSchedule, fixedRate, payer)
            #Will use 5 standard deviations to integrate over since this capture approxx 99% of the distribution
            resetSd       = 5*np.sqrt(self.varFwd(0,TR))
            expirySd      = 5*np.sqrt(self.varFwd(0,expiry))
            return self.ZCB(expiry)*integrate.dblquad(toIntegrate, -5*expirySd, 5*expirySd, -5*resetSd, 5*resetSd)[0]
        else:
            print('Error: Expiry date is before first reset date')
            return None

    def swaption(self, expiry, fixedSchedule, floatSchedule, fixedRate, payer=True):
        TR = floatSchedule[floatSchedule>=expiry-0.5][0] #First reset date
        TRp1 = floatSchedule[floatSchedule>=expiry-0.5][1] #second reset date
        if (TR==0) and (TR==expiry):
            return np.maximum(self.swap(0, fixedSchedule, floatSchedule, fixedRate),0)
        else:
            return self.swaptionIntegration(expiry,fixedSchedule,floatSchedule, fixedRate, payer)
        

    def swaptionSC(self, time, expiry, fixedSchedule, floatingSchedule, fixedRate, payer=True):

        floatingSchedule = floatingSchedule[floatingSchedule>expiry-0.5]
        if len(floatingSchedule) == 0:
            TR=10
        else:
            TR=floatingSchedule[0]

        fixedSchedule = fixedSchedule[fixedSchedule>expiry-1]

        #finding rBar
        rbar   = fsolve(func=lambda r: self.rbarhelper(t=0, expiry=expiry, rbar=r, fixedSchedule=fixedSchedule, floatingSchedule = floatingSchedule, fixedRate=fixedRate), x0=self.init)[0]
        sum = 0

        for Si in fixedSchedule[1::]:
            ci = fixedRate
            if Si == fixedSchedule[-1]:
                ci += 1

            Xi = self.ZCB(Si-expiry,time=0,initRate=rbar)
            if payer:
                put = self.ZCBPut(time, expiry, Si, Xi)
                sum += ci*put
            elif not payer:
                call = self.ZCBCall(time, expiry, Si, Xi)
                sum += ci*call

        return sum