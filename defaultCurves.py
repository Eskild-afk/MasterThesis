import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
timeStamp  = np.array([0, 180,360,720,1080,1440,1800,2520,3600,5400,7200,10800])/365
# CDSspreadI = np.array([172.9, 173.7, 155.5, 152.3, 144.0, 142.3, 148.3, 156.4])/10000 #Danske Bank CDS
CDSspreadI = np.array([80,91,105,125,146,163,181,200,216,219,222,227])/10000
CDSspreadC = np.array([20,24,29,48,72,99,126,159,183,195,202,213])/10000
R = 0.4
l = 0.03

lI = lambda x: CDSspreadI[np.where(timeStamp > x )][0]*x/(1-R)
lC = lambda x: CDSspreadC[np.where(timeStamp > x )][0]*x/(1-R)

SI = interpolate.interp1d(timeStamp, np.exp(-CDSspreadI*timeStamp/(1-R)), fill_value='extrapolate')
SC = interpolate.interp1d(timeStamp, np.exp(-CDSspreadC*timeStamp/(1-R)), fill_value='extrapolate')


def li(t):
        try:
            return CDSspreadI[np.where(timeStamp > t )][0]
        except:
            return CDSspreadI[0]
    
def lc(t):
        try:
            return CDSspreadC[np.where(timeStamp > t )][0]
        except:
            return CDSspreadC[0]
        
def SI(t):
    return np.exp(-li(t)*t/(1-R))

def SC(t):
    return np.exp(-lc(t)*t/(1-R))

def QI(t):
    return np.exp(-li(t)*t/(1-R))

def QC(t):
    return np.exp(-lc(t)*t/(1-R))


def CVA(time, dt,PE):
    sum = 0
    for i in range(len(time)):
        sum += PE[i]*SI(time[i])*(QC(time[i])-QC(time[i]-dt))
    return -sum
def DVA(time, dt, NE):
    sum = 0
    for i in range(len(time)):
        sum += NE[i]*SC(time[i])*(QI(time[i])-QI(time[i]-dt))
    return -sum


# sns.lineplot(x=np.arange(0, 15, 1/365), y=[lI(t) for t in np.arange(0, 15, 1/365)], label='Hazard Rates I')
# sns.lineplot(x=np.arange(0, 15, 1/365), y=[lC(t) for t in np.arange(0, 15, 1/365)], label='Hazard Rates C')
# plt.show()

# sns.lineplot(x=np.arange(0, 15, 1/365), y=[np.exp(-lI(t)) for t in np.arange(0, 15, 1/365)], label='Survival I')
# sns.lineplot(x=np.arange(0, 15, 1/365), y=[np.exp(-lC(t)) for t in np.arange(0, 15, 1/365)], label='Survival C')
# # sns.scatterplot(x=timeStamp, y=np.exp(-cf[0]*timeStamp), label='curvefit')
# plt.show()
