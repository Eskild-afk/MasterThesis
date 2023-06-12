import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.integrate import quad
# timeStamp  = np.array([0, 180,360,720,1080,1440,1800,2520,3600,5400,7200,10800])/365 #Article time stamp
timeStamp = np.array([0.5,1,2,3,4,5,7,10])
CDSspreadI = np.array([34.8,  41.6 ,  51.5,  61.1,  71.8,  83.6,  95.9, 104.9])/10000 #Danske Bank CDS
CDSspreadC = np.array([172.9, 173.7, 155.5, 152.3, 144.0, 142.3, 148.3, 156.4])/10000 #Deutche Bank CDS
# CDSspreadI = np.array([80,91,105,125,146,163,181,200,216,219,222,227])/10000 # From Article
# CDSspreadC = np.array([20,24,29,48,72,99,126,159,183,195,202,213])/10000 # From Article
R = 0.4
l = 0.03

# lI = lambda x: CDSspreadI[np.where(timeStamp > x )][0]*x/(1-R)
# lC = lambda x: CDSspreadC[np.where(timeStamp > x )][0]*x/(1-R)

# SI = interpolate.interp1d(timeStamp, np.exp(-CDSspreadI*timeStamp/(1-R)), fill_value='extrapolate')
# SC = interpolate.interp1d(timeStamp, np.exp(-CDSspreadC*timeStamp/(1-R)), fill_value='extrapolate')


def lI(t):
        if any(timeStamp>=t):
            return CDSspreadI[np.where(timeStamp >= t )][0]/(1-R)
        else:
            return CDSspreadI[-1]/(1-R)
    
def lC(t):
        if any(timeStamp>=t):
            return CDSspreadC[np.where(timeStamp >= t )][0]/(1-R)
        else:
            return CDSspreadC[-1]/(1-R)
        
def SI(t):
    # return np.exp(-quad(lambda x: lI(x), 0, t)[0])
    return np.exp(-lI(t)*t)

def SC(t):
    # return np.exp(-quad(lambda x: lC(x), 0, t)[0])
    return np.exp(-lC(t)*t)

def QI(t):
    return 1-SI(t)

def QC(t):
    return 1-SC(t)


def CVA(time, dt,PE):
    sum = 0
    for i in range(1,len(time)):
        sum += PE[i]*SI(time[i])*(QC(time[i])-QC(time[i-1]))
    return -sum
def DVA(time, dt, NE):
    sum = 0
    for i in range(1,len(time)):
        sum += NE[i]*SC(time[i])*(QI(time[i])-QI(time[i-1]))
    return -sum

def FCA(time, PE, s):
    sum = 0
    for i in range(1,len(time)):
        sum += s*SI(time[i])*SC(time[i])*PE[i]*(time[i]-time[i-1])
    return -sum

def FBA(time, NE, s):
    sum = 0
    for i in range(1,len(time)):
        sum += s*SI(time[i])*SC(time[i])*NE[i]*(time[i]-time[i-1])
    return -sum


def ColVa(time, Col, s):
    sum = 0
    for i in range(1,len(time)):
        sum += s*SI(time[i])*SC(time[i])*Col[i]*(time[i]-time[i-1])
    return -sum

fig, ax = plt.subplots()
fig.set_size_inches(15,8)
sns.lineplot(x=np.arange(0, 10, 1/365), y=[lI(t)*10000 for t in np.arange(0, 10, 1/365)], label='CDS Spreads for Investor')
sns.lineplot(x=np.arange(0, 10, 1/365), y=[lC(t)*10000 for t in np.arange(0, 10, 1/365)], label='CDS Spreads for Counterparty')
ax.set_xlabel('Tenor (Years)', fontname="Times New Roman", fontsize = 28)
ax.set_ylabel('Spread (bps)', fontname="Times New Roman", fontsize = 28)
ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
ax.axhline(y=0, color='k', alpha = 0.25)
plt.grid(alpha = 0.25)
plt.xticks(np.arange(0, 16, 1))
plt.xlim(0,10)
ax.xaxis.set_label_coords(0.5, -0.11)
ax.yaxis.set_label_coords(-0.1, 0.5)
plt.legend(frameon = False, fontsize = 18, loc='upper right')
plt.savefig(f'./Graphs/CDS_Spreads.png', bbox_inches='tight')



fig, ax = plt.subplots()
fig.set_size_inches(15,8)
sns.lineplot(x=np.arange(0, 10, 1/365), y=[QI(t)*100 for t in np.arange(0, 10, 1/365)], label='Default probability for Investor')
sns.lineplot(x=np.arange(0, 10, 1/365), y=[QC(t)*100 for t in np.arange(0, 10, 1/365)], label='Default probability for Counterpart')
ax.set_xlabel('Time (Years)', fontname="Times New Roman", fontsize = 28)
ax.set_ylabel('Default probability (%)', fontname="Times New Roman", fontsize = 28)
ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
ax.axhline(y=0, color='k', alpha = 0.25)
plt.grid(alpha = 0.25)
plt.xlim(0,10)
plt.xticks(np.arange(0, 11, 1))
ax.xaxis.set_label_coords(0.5, -0.11)
ax.yaxis.set_label_coords(-0.1, 0.5)
plt.legend(frameon = False, fontsize = 18, loc='upper right')
plt.savefig(f'./Graphs/Default Probability.png', bbox_inches='tight')