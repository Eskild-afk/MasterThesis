from array import array
import numpy as np
from dynamics import Dynamic
#from pandas._libs.hashtable import Vector
from helpers import * 

class NelsonSiegel(Dynamic):
    """
    Class for 3-factor Nelson Siegel
    """

    def __init__(
            self, 
            Lambda: float or np.array or None, 
            sigmaL: float or np.array or None, 
            sigmaS: float or None, 
            sigmaC: float or None, 
            initial: np.array or None #Xt
        ) -> None:

        super().__init__(initial)

        self.Xt = initial
        self.Lambda = Lambda  # Lambda with cap since lambda is reserved
        self.sigmaL = sigmaL
        self.sigmaS = sigmaS
        self.sigmaC = sigmaC

    def A(self, t, T):
        bracket1 = 2 * (T-t) * self.Lambda - 3 + 4 * np.exp(-(T-t) * self.Lambda) - np.exp(
            -2 * (T-t) * self.Lambda)
        bracket2 = (1 + (- 1 - 2 * (T - t)**2 * np.square(self.Lambda) - 2 * self.Lambda * (T - t))*np.exp(-2*self.Lambda*(T-t)))
        return (np.square(self.sigmaL) * np.power((T-t), 3)) / 6 + np.square(self.sigmaS) / (
                4 * np.power(self.Lambda, 3)) * bracket1 + bracket2/(4*np.power(self.Lambda,3))

    def Avector(self):
        return np.array([[-self.A(t=t, T=T)/t for t in self.tList]])

    def B(self, t, T):
        return np.array([-(T-t), (np.exp(-self.Lambda * (T-t)) - 1) / self.Lambda, (T-t) * np.exp( - self.Lambda *(T-t))]).reshape(3,1)

    def Bmatrix(self):
        return np.array([-self.B(t=t, T=T)/t for t in self.tList])

    def oneStep(self, t, stepfrom, stepsize, fwd = 0, Z=None):
        z = np.random.normal(loc=0,scale=1, size=3).reshape(3,1)

        KQ = np.array([
            [0, 0, 0],
            [0, self.Lambda, -self.Lambda],
            [0,0,self.Lambda]
            ]).reshape(3,3)

        ThetaQ = np.array([[0],[0],[0]]).reshape(3,1)

        Sigma = np.array([
            [self.sigmaL, 0, 0],
            [0, self.sigmaS, 0],
            [0,0,self.sigmaC]
            ]).reshape(3,3)
        if fwd == 0:
            dX = np.matmul(KQ,ThetaQ - stepfrom.reshape(3,1))*stepsize + np.matmul(Sigma, np.sqrt(stepsize)*z)
        else:
            dX = np.matmul(KQ,ThetaQ - stepfrom.reshape(3,1))*stepsize + np.matmul(Sigma, np.sqrt(stepsize)*(z-Sigma @ self.B(t, fwd).reshape(3,1)))

        return (stepfrom.reshape(3,1)+dX).flatten()

    def ZCB(self, t, T, initial = np.repeat(None,3)):
        if initial.any() == None:
            initial = self.Xt
        zcb = np.exp( - self.A(t,T) - np.matmul( self.B(t,T).T , initial )[0])

        return zcb

    def swap(self, t, S:np.array, T:np.array, K, initRate=np.array([None,None,None]), payer=True):
        '''
        t: time the swap should be priced at given the rate r(t)
        S: vector of of fixed schedule reset dates S0..Sm
        T: vector of of floating schedule reset dates T0..Tn
        initRate: initial rate of the swap
        payer: True if payer, False if receiver
        '''

        if (initRate==None).any():
            initRate = self.Xt

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

    def swapextended(self, t, S:np.array, T:np.array, K, floatRate, schedule, initRate=None, payer=True):
        if (initRate==None).any():
            initRate = self.Xt
        
        if t >= S[-1]:
            return 0
        else: 
            TR = T[T >= t-0.5][0]
            TRp1 = T[T >= t-0.5][1]
            fixedleft = S[S >= t - 1]
            rTR = floatRate[np.where(schedule>=TR)][0]
            r = floatRate[find_nearest(schedule,t)]
            # r = floatRate[np.where(schedule==t)][0]
            sum = 0
            for i in fixedleft[1::]:
                c=K
                if i == fixedleft[-1]:
                    c+=1
                sum += c*self.ZCB(t, i, initial=r)
            if t > TR:
                return self.ZCB(t, TRp1, initial=r)/self.ZCB(TR, TRp1, initial=rTR)-sum
            else: 
                return self.ZCB(t, TR, initial=r)-sum