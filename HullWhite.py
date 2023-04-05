from dynamics import Dynamic
from scipy.optimize import minimize, fsolve
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy import integrate
class HullWhite(Dynamic):
    '''
    The dynamic is (mean+rev*r)dt + vol*gamma(T)*dW
    Simple single factor Vasicek model with piecewise constant volatility
    initial is the starting rate
    mean is the mean reverting parameter
    reversion is how quickly it returns to the mean
    volatility doesn't need an explanation
    '''
    def __init__(
            self, 
            initial,    #r/r_0
            reversion,  #beta
            volatility, #sigma
            Gamma,      #PieceVolParams
            b:np.array, #Params for the forward curve
            tau:np.array#Params for the forward curve
        )->None:
        
        super().__init__(initial) 
        self.rev    = reversion
        self.vol    = volatility
        self.Gamma  = Gamma
        self.b      = b
        self.tau    = tau
    
    def yieldCurve(self, t):
        '''
        Nelson-Siegel-Svenson model yield curve
        '''
        if t==0:
            return self.b[0]+self.b[1]
        
        nss =  0
        nss += self.b[0]
        nss += self.b[1] * ((1 - np.exp(-t / self.tau[0])) / (t / self.tau[0]))
        nss += self.b[2] * (((1 - np.exp(-t / self.tau[0])) / (t / self.tau[0])) - np.exp(-t / self.tau[0]))
        nss += self.b[3] * (((1 - np.exp(-t / self.tau[1])) / (t / self.tau[1])) - np.exp(-t / self.tau[1]))

        return  nss/100

    
    def forwardCurve(self, t, b:np.array, tau:np.array):
        '''
        b and tau are the parameters of the NelsonSiegelSvenson model
        Can be found on:
        https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html
        Note that these parameters return the rate in percentage, so we divide by 100
        b is vector of 4 values, and tau is vector of 2 values.
        '''

        fwd = b[0]+b[1]/np.exp(t/tau[0])+b[2]*t/np.exp(t/tau[0])/tau[0]+b[3]*t/np.exp(t/tau[1])/tau[1]
        
        return fwd/100
    
    def forwardCurveDerivative(self, t, b:np.array, tau:np.array):
        '''
        Derivative of the forward curve with respect to time
        '''
        fwdDiff = b[2]/(tau[0]*np.exp(t/tau[0]))+b[3]/(tau[1]*np.exp(t/tau[1]))-b[1]/(tau[0]*np.exp(t/tau[0]))-(t*b[2])/(tau[0]**2*np.exp(t/tau[0]))-(t*b[3])/(tau[1]**2*np.exp(t/tau[1]))

        return fwdDiff/100
    
    def theta(self, t):
        '''
        Mean reverting function
        See Brigo & Mercurio, 2006 eq. (3.34)
        '''
        theta =  0
        theta += self.forwardCurveDerivative(t, self.b, self.tau)
        theta += self.rev*self.forwardCurve(t, self.b, self.tau)
        theta += self.vol**2/(2*self.rev)*np.square(1-np.exp(-self.rev*t))

        return theta
    
    def alpha(self, t):
        '''
        Alpha function
        See Brigo & Mercurio, 2006 eq. (3.36)
        '''

        alpha = 0
        alpha += self.forwardCurve(t, self.b, self.tau)
        alpha += self.vol**2/(2*self.rev**2)*(1-np.exp(-2*self.rev*t))

        return alpha
    
    def oneStep(self, t, stepfrom, stepsize, fwd=0):
        '''
        One step of the simulation
        '''
        
        if fwd==0:
            dr = (self.theta(t)-self.rev*stepfrom)*stepsize + self.vol*np.sqrt(stepsize)*np.random.normal()
        
        else: # for now it is the same
            dr = (self.theta(t)-self.rev*stepfrom)*stepsize - self.M(t, t+stepsize, fwd) + self.vol*np.sqrt(stepsize)*np.random.normal()
        
        return stepfrom+dr
    
    def expectedRate(self, t):
        er = self.init*np.exp(-self.rev*t)+self.alpha(t)-self.alpha(0)*np.exp(-self.rev*t)
        return er

    def variance(self, t):
        var = self.vol**2/(2*self.rev)*(1-np.exp(-2*self.rev*t))
        return var
    
    def B(self, t,T):
        B = (1-np.exp(-self.rev*(T-t)))/self.rev
        return B
    
    def V(self, t,T):
        V =  T-t
        V += 2/self.rev*np.exp(-self.rev*(T-t))
        V -= 1/(2*self.rev)*np.exp(-2*self.rev*(T-t))
        V -= 3/(2*self.rev)

        return (self.vol**2)/(self.rev**2)*V
    
    def A(self, t,T):
        A =  self.B(t,T)*self.forwardCurve(t, self.b, self.tau)
        A -= self.vol**2/(4*self.rev)*(1-np.exp(-2*self.rev*(t)))*np.square(self.B(t,T))
        return A

    def marketZCB(self, t):
        '''
        Calcculates the market ZCB price given an expiry t, i.e P(0,t)
        '''
        zcb = np.exp(-self.yieldCurve(t)*t)
        return zcb

    def C(self, t, T, S):
        # cov = self.vol**2*np.exp(-self.rev*(T+S))*(np.exp(-2*self.rev*T)-np.exp(-2*self.rev*t))
        cov = self.vol*np.exp(-self.rev*(T+S))*np.exp(2*(T-t))/2
        return np.array([self.V(t,T), cov, cov, self.V(t,S)]).reshape(2,2)

    def ZCB(self, t,T, initRate=None):
        if initRate==None:
            initRate = self.init
        zcb = np.exp(self.A(t,T)-initRate*self.B(t,T))*self.marketZCB(T)/self.marketZCB(t)
        return zcb
    
    def M(self, s,t,T):
        '''
        s: time now
        t: time to
        T: forward measure
        '''
        M =  0
        M += self.vol**2/(self.rev**2)*(1-np.exp(-self.rev*(t-s)))
        M -= self.vol**2/(2*self.rev**2)*(np.exp(-self.rev*(T-t))-np.exp(-self.rev*(T+t-2*s)))

        return M
    
    def ZCBoption(self, t,T,S,X, initRate=None, call=True):
        '''
        Zero Coupon Bond option call eq(3.40)
        '''
        if initRate==None:
            initRate = self.init

        if call:
            w=1
        else:
            w=-1
        sigmap = self.vol*np.sqrt((1-np.exp(-2*self.rev*(T-t)))/(2*self.rev))*self.B(T,S)
        h      = np.log(self.ZCB(t,S,initRate)/(self.ZCB(t,T,initRate)*X))/sigmap+sigmap/2

        norm1 = norm.cdf(w*h)
        norm2 = norm.cdf(w*(h-sigmap))

        return w*(self.ZCB(t,S,initRate)*norm1-self.ZCB(t,T,initRate)*X*norm2)
    
    def swap(self, t, S:np.array, T:np.array, K, initRate=None, payer=True):
        '''
        t: time the swap should be priced at given the rate r(t)
        S: vector of of fixed schedule reset dates S0..Sm
        T: vector of of floating schedule reset dates T0..Tn
        initRate: initial rate of the swap
        payer: True if payer, False if receiver
        '''
        if initRate==None:
            initRate = self.init
        
        if t >= T[-1]:
            return 0

        if payer:
            w=1
        else:
            w=-1
        
        # First reset date:
        T  = T[T>=t-0.5]
        S  = S[S>=t-1]

        sum = 0
        for Si in S[1::]:
            sum += K*self.ZCB(t,Si,initRate)
        
        return w*(self.ZCB(t,T[0],initRate)-self.ZCB(t,T[-1],initRate)-sum)
    

    def rbarhelper(self, t, expiry, rbar, S, T, K):
        sum = 0

    
        for Si in S[1::]:
            c=K     
            if Si == S[-1]:
                c+=1    
            
            sum += c*self.ZCB(expiry, Si, initRate=rbar)
        return 1-sum
    

    def swaption(self, t, Te, S:np.array, T:np.array, K, initRate=None, payer=True):
        
        if initRate==None:
            initRate = self.init

        if t>=Te:
            return self.swap(t, S, T, K, initRate, payer)
        
        if Te>= T[-1]:
            return 0
        
        T = T[T>=Te-0.5]
        S = S[S>=Te-1]
        
        firstResetFloat = T[0]
        firstResetFix   = S[0]

        T[0] = np.maximum(firstResetFloat, Te)
        S[0] = np.maximum(firstResetFix, Te)

        if t==S[0]:
            return np.maximum(self.swap(t, S, T, K, initRate, payer),0)
        

        sum = 0
        # if Te <= S[0]:
        rbar = fsolve(
            func = lambda x : self.rbarhelper(t=t, expiry=Te, rbar=x, S=S, T=T, K=K),
            x0  = initRate
        )[0]
        for Si in S[1::]:
            ci = K
            if Si == S[-1]:
                ci += 1

            Xi = self.ZCB(Te, Si, initRate=rbar)
            sum += ci*self.ZCBoption(t, Te, Si, Xi,initRate=initRate, call=(not payer))
            
        return sum

