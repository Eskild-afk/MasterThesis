import pandas as pd
import numpy as np
from HullWhite import HullWhite
from scipy.optimize import fsolve
S = np.arange(0,11)
T = np.arange(0,10.5,0.5)
beta = np.array([1.879892, 0.579211, 	3.993992, 1.445091])
tau  = np.array([ 16.633491, 	0.319680])
reversion=0.08670264780833303 #0.13949636660880768 
volatility=0.013928489964789946 #0.017793899652989272
HW = HullWhite(initial=0.02459103, reversion=reversion, volatility=volatility, b=beta, tau=tau)
K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]


from joblib import load, Parallel, delayed, cpu_count, dump
import seaborn as sns
import time
EPE = load('./SimulationData/PE_10Y_Swap_N=100000_dt=365.joblib')
ENE = load('./SimulationData/NE_10Y_Swap_N=100000_dt=365.joblib')

def worker(Te):
    print(Te, end='                                                              \r')
    return HW.swaption(0,Te,S,T,K, payer=True), HW.swaption(0,Te,S,T,K, payer=False)
Ncpu = 16#int(cpu_count()/2)
time = np.arange(0,10+1/12,1/12)
AnalyticalEESwap10Y = Parallel(n_jobs=Ncpu)(delayed(worker)(Te) for Te in time)

PE = []
NE = []

for i in range(len(time)):
    PE.append(AnalyticalEESwap10Y[i][0])
    NE.append(-AnalyticalEESwap10Y[i][1])


dump(PE, './SimulationData/AnalyticalPESwap10Y.joblib')
dump(NE, './SimulationData/AnalyticalNESwap10Y.joblib')

# def worker(Te):
#     return Te, HW.swaption(0,Te,S,T,K, payer=False)
# AnalyticalNESwap10Y = Parallel(n_jobs=cpu_count())(delayed(worker)(Te) for Te in np.arange(0,10+1/365,1/365))
# dump(AnalyticalNESwap10Y, './SimulationData/AnalyticalNESwap10Y.joblib')