import numpy as np
from dynamics import *
from helpers import *

def parSwapRate(time:float, fixedSchedule:np.array, floatingSchedule:np.array,  floatingRate:np.array, floatingTimeStamp:np.array, model:Dynamic):
    top = 0
    bot = 0
    rate = floatingRate[np.where(floatingTimeStamp==time)][0]
    for j in range(1, len(floatingSchedule)):
        Tj  = floatingSchedule[j]   # T_j
        Tjm = floatingSchedule[j-1] # T_{j-1}
        if Tj >= time:
            if Tjm <= time:
                nt = find_nearest(floatingTimeStamp, Tjm)  #nearest time (by construction it should be exactly equal but decimals and such can fuck it up)
                rjm = floatingRate[np.where(floatingTimeStamp==nt)][0]   #rate at time Tjm
                F = (1/model.ZCB(duration=Tj-Tjm, initRate=rjm)-1)/(Tj-Tjm) # ZCB functionen er ændret så den tager duration, time og initRate

            else: 
                F = model.forward_rate(time=time, start=Tjm, end=Tj, initRate=rate)

            D = model.ZCB(duration=Tj-time, initRate=rate)
            top += D*F*(Tj-Tjm)
        
    for i in range(1, len(fixedSchedule)):
        Si  = fixedSchedule[i]   # S_j
        Sim = fixedSchedule[i-1] # S_{j-1}

        if Si >= time:
            D = model.ZCB(duration=Si-time, initRate=rate)
            bot += D*(Si-Sim)

    return top/bot


def payerSwap(time:float, fixedSchedule:np.array, floatingSchedule:np.array, fixedRate:float, floatingRate:np.array, floatingTimeStamp:np.array, model:Dynamic):
    '''
    Currently only works where both schedule are the same and have the same stepsize

    fixedSchedule is the schedule time in which the fixed rates are paid
    fixedRate is the rate which is paid
    floatingSchedule is the schdule in which the floating rate is received
    floatingRate is the rate recived
    floatingTimeStamp are the coherent time stamps of the floating rate observations
    model is which model is used to calculate the ZCB
    '''

    if fixedSchedule[0] != floatingSchedule[0]:
        raise Exception("Floating and fixed schedule are not the same to start with")

    if fixedSchedule[-1] != floatingSchedule[-1]:
        raise Exception("Floating and fixed schedule are not the same to end with")

    if time >= floatingSchedule[-1]:
        return 0
    
    r = floatingRate[np.where(floatingTimeStamp==time)][0] # rate at time t

    #floating part
    # print(f'Time={time}')
    floatingPart = 0
    for j in range(1, len(floatingSchedule)):
        Tj  = floatingSchedule[j]                                # T_j
        Tjm = floatingSchedule[j-1]                              # T_{j-1}
        
        if Tj >= time:
            if Tjm <= time:
                nt = find_nearest(floatingTimeStamp, Tjm)                   #nearest time (by construction it should be exactly equal but decimals and such can fuck it up)
                rjm = floatingRate[np.where(floatingTimeStamp==nt)][0]      #rate at time Tjm
                F = (1/model.ZCB(duration=Tj-Tjm, time=time, initRate=rjm)-1)/(Tj-Tjm)
                
            else:
                F = model.forward_rate(time, Tjm, Tj, initRate=r)
                
            D = model.ZCB(duration=Tj-time, time=time, initRate=r)
            floatingPart += D*F*(Tj-Tjm)
    
    #fixed part
    fixedPart = 0
    for i in range(1, len(fixedSchedule)):
        Si  = fixedSchedule[i]   # S_j
        Sim = fixedSchedule[i-1] # S_{j-1}

        if Si >= time:
            D = model.ZCB(duration=Si-time, time=time, initRate=r)
            fixedPart += D*(fixedRate)*(Si-Sim)
    
    return floatingPart - fixedPart

def payerSwaption(time:float, expiry:float, fixedSchedule: np.array, floatingSchedule: np.array, fixedRate:float, floatingRate:np.array, floatingTimeStamp:np.array, model:Dynamic):
    '''
    '''
    PSR = parSwapRate(expiry, fixedSchedule, floatingSchedule, floatingRate, floatingTimeStamp, model)
    A = 0
    r = floatingRate[np.where(floatingTimeStamp==time)][0]
    if time <= expiry:
        for i in range(1, len(fixedSchedule)):
            Si  = fixedSchedule[i]   # S_j
            Sim = fixedSchedule[i-1] # S_{j-1}

            if Si >= time:
                D = model.ZCB(duration=Si-time, time=time, initRate=r)
                A += D*(Si-Sim)
        return A*np.maximum(PSR-fixedRate,0)
    
    else:

        return payerSwap(time, fixedSchedule, floatingSchedule, fixedRate, floatingRate, floatingTimeStamp, model)