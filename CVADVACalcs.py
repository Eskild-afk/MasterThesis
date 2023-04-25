from joblib import load
from defaultCurves import *
from HullWhite import *

beta = np.array([1.879892, 0.579211, 	3.993992, 1.445091])
tau  = np.array([ 16.633491, 	0.319680])
reversion=0.13949636660880768 
volatility=0.017793899652989272
# HW = HullWhite(initial=0.02459103, reversion=0.03, volatility=0.00200, Gamma=1000, b=beta, tau=tau)
HW = HullWhite(initial=0.02459103, reversion=reversion, volatility=volatility, b=beta, tau=tau)
time = HW.create_path(10, 1/4)

NE = load('./SimulationData/NE_10Y_Swap_N=5000_dt=4.joblib')

PE = load('./SimulationData/PE_10Y_Swap_N=5000_dt=4.joblib')
dt = 1/4
sum = 0
for i in range(len(time)):
    sum += PE[i]*SC(time[i])*lC(time[i])*QC(time[i]+dt)

CVA = -sum

sum = 0
for i in range(len(time)):
    sum += NE[i]*SI(time[i])*lC(time[i])*QC(time[i]+dt)

DVA = -sum

print(CVA, DVA, sum)