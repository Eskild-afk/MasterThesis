# Creating presaved paths
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, cpu_count, dump
from scipy.optimize import fsolve
from HullWhite import HullWhite
import time as timer

import matplotlib.font_manager

beta = np.array([1.879892, 0.579211, 	3.993992, 1.445091])
tau  = np.array([ 16.633491, 	0.319680])
reversion=0.13949636660880768 
volatility=0.017793899652989272
# HW = HullWhite(initial=0.02459103, reversion=0.03, volatility=0.00200, Gamma=1000, b=beta, tau=tau)
HW = HullWhite(initial=0.02459103, reversion=reversion, volatility=volatility, Gamma=1000, b=beta, tau=tau)
preSavedSims = []
dt=1/365
sims = 5000
def worker(i):
        float = HW.create_path(dt,30, seed=i)[1]
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return float


results = Parallel(n_jobs=np.minimum(cpu_count(),sims))(delayed(worker)(i) for i in range(sims))

dump(results, f'SimulationData/HW_30yrs_1D_N={sims}.joblib')