import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
timeStamp  = np.array([0, 180,360,720,1080,1440,1800,2520,3600,5400,7200,10800])/365
CDSspreadI = np.array([80,91,105,125,146,163,181,200,216,219,222,227])/10000
CDSspreadC = np.array([20,24,29,48,72,99,126,159,183,195,202,213])/10000
R = 0.4
l = 0.03

SI = interpolate.interp1d(timeStamp, np.exp(-CDSspreadI*timeStamp/(1-R)), fill_value='extrapolate')
SC = interpolate.interp1d(timeStamp, np.exp(-CDSspreadC*timeStamp/(1-R)), fill_value='extrapolate')

lI = interpolate.CubicSpline(timeStamp, CDSspreadI)
lC = interpolate.CubicSpline(timeStamp, CDSspreadC)
sns.scatterplot(x=timeStamp, y=np.exp(-CDSspreadI*timeStamp/(1-R)), label='I')
sns.scatterplot(x=timeStamp, y=np.exp(-CDSspreadC*timeStamp/(1-R)), label='C')
sns.lineplot(x=np.arange(0, 40, 1/365), y=SI(np.arange(0, 40, 1/365)), label='LinearI')
sns.lineplot(x=np.arange(0, 40, 1/365), y=SC(np.arange(0, 40, 1/365)), label='LinearC')
sns.lineplot(x=np.arange(0, 40, 1/365), y=[np.exp(-lI(t)*t/(1-R)) for t in np.arange(0, 40, 1/365)], label='SplineI')
sns.lineplot(x=np.arange(0, 40, 1/365), y=[np.exp(-lC(t)*t/(1-R)) for t in np.arange(0, 40, 1/365)], label='SplineC')
# sns.scatterplot(x=timeStamp, y=np.exp(-cf[0]*timeStamp), label='curvefit')
plt.show()
