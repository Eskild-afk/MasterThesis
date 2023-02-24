import numpy as np
from dynamics import *


def payerSwap(time:float, fixedSchedule:np.array, floatingSchedule:np.array, fixedRate:float, floatingRate:np.array, floatingTimeStamp:np.array, stepsize:float, model:Dynamic):
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

    #Firstly, we are only interested in the remaining payments, so we cut off schedules that have already happed
    floatingSchedule = floatingSchedule[floatingSchedule>time] 
    fixedSchedule    = fixedSchedule[fixedSchedule>time]    
    
    #floating part
    floatingPart = 0
    for j in floatingSchedule:
        D = model.ZCB(duration=(j-time)*stepsize, initRate=floatingRate[np.where(floatingTimeStamp==time)][0])
        F = model.forward_rate(time*stepsize, (j-1)*stepsize, j*stepsize, initRate=floatingRate[np.where(floatingTimeStamp==time)][0])*stepsize
        floatingPart += D*F
    
    #fixed part
    fixedPart = 0
    for i in fixedSchedule:
        D = model.ZCB(duration=(i-time)*stepsize, initRate=floatingRate[np.where(floatingTimeStamp==time)][0])
        fixedPart += D*(fixedRate*stepsize)

    return floatingPart - fixedPart

def payerSwaption(time:float, fixedSchedule):
    pass