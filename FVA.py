import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, cpu_count, dump, load
from scipy.optimize import fsolve
from HullWhite import HullWhite
import time as timer
from helpers import *
import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths='C:\Windows\Fonts', fontext='ttf')
from defaultCurves import *
import sys

beta = np.array([1.879892, 0.579211, 	3.993992, 1.445091])
tau  = np.array([ 16.633491, 	0.319680])
reversion=0.08670264780833303 #0.13949636660880768
volatility=0.013928489964789946 #0.017793899652989272
HW = HullWhite(initial=0.02459103, reversion=reversion, volatility=volatility, b=beta, tau=tau)

nCPU=int(cpu_count()/1-0)
#Setting up tenor
T=np.arange(0,10+0.5,0.5)
S=np.arange(0,11,1)

def BP(bp):
    '''Converts basis points to decimal'''
    return bp/10000

sm = BP(25) # spread minus
sp = BP(100) # spread plus

if False: #If it should use average between fundings
    avg = 0.5*(sm+sp)
    sm = avg
    sp = avg
# Other settings
dt   = 1/12
sims = 5000
total_time = timer.time()

KVM=0 # Threshold for VM
KIM=0 # Threshold for IM
MTA=0 # Minimum Transfer Amount
lag=2/365 # Lookback lag
#10Y Payer Swap Exposure
print(f'Simulation started with dt=1/{int(1/dt)} and N={sims}')
if True:
    print('10Y Payer Swap Exposure')
    start = timer.time()
    
    #Constructing time grid
    time = np.arange(0,10+dt,dt)
    time = time[np.where(time<=10)]

    #Constructing lagged grid
    if dt == 1/365:
        lagged_time = time[0:len(time)-2]
        ttso = time   #times to simulate on
    else:
        lagged_time = (time-lag)[1::]
        ttso = np.insert(time, np.arange(1,len(time)), lagged_time)

    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]

    def worker(i):
        float = HW.create_path(dt, 10,fwd=0,seed=i)[1]
        # D = np.insert(np.exp(-float[:-1:]*time[1::]),0,1)
        
        swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
        dfloat = np.where(swap>=0, float+sp, float+sm)
        D = np.cumprod(np.insert(np.exp(-dfloat[:-1:]*dt),0,1))
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return D*np.maximum(swap,0), D*np.minimum(swap,0)

    print('Starting parallel processing with {} cores'.format(nCPU))
    results = Parallel(n_jobs=nCPU)(delayed(worker)(i) for i in range(sims))
    # results = [worker(i) for i in range(sims)]
    print('Finished parallel processing')
    PE = np.zeros(len(time))
    NE = np.zeros(len(time))
    print('Starting sequential processing')
    for i in range(len(results)):
        PE += results[i][0]
        NE += results[i][1]
    print(f'Finished sequential processing in {timer.time()-start:.2f}s')
    print('Creating graph')

    # discounting = np.array([HW.ZCB(0,t) for t in time])
    discounting = 1
    sigmaP = 0
    sigmaM = 0
    for H in results:
        sigmaP += (discounting*(H[0]-PE/sims))**2
        sigmaM += (discounting*(H[1]-NE/sims))**2
    sigmaP = np.sqrt(sigmaP/(sims-1))
    sigmaM = np.sqrt(sigmaM/(sims-1))

    HPUB = discounting*PE/sims+3*sigmaP/np.sqrt(sims)
    HPLB = discounting*PE/sims-3*sigmaP/np.sqrt(sims)
    HMUB = discounting*NE/sims+3*sigmaM/np.sqrt(sims)
    HMLB = discounting*NE/sims-3*sigmaM/np.sqrt(sims)
    # sigmaP = 0
    # sigmaM = 0
    # EPE =  load("./SimulationData/PE_10Y_Swap_N=100000_dt=365.joblib")
    # ENE =  load("./SimulationData/NE_10Y_Swap_N=100000_dt=365.joblib")
    # for H in results:
    #     sigmaP += (discounting*(H[0]-EPE))**2
    #     sigmaM += (discounting*(H[1]-ENE))**2
    # sigmaP = np.sqrt(sigmaP/(sims-1))
    # sigmaM = np.sqrt(sigmaM/(sims-1))

    # HPUB = discounting*EPE+3*sigmaP/np.sqrt(sims)
    # HPLB = discounting*EPE-3*sigmaP/np.sqrt(sims)
    # HMUB = discounting*ENE+3*sigmaM/np.sqrt(sims)
    # HMLB = discounting*ENE-3*sigmaM/np.sqrt(sims)

    plt.rc('font',family='Times New Roman')
    float_x = np.arange(0.5,10.5,0.5)
    float_y = np.full((len(float_x),), -5.6)
    fix_x = np.arange(1,11)
    fix_y = np.full(len(fix_x), -5)

    fig, ax = plt.subplots()
    sns.lineplot(x=time, y=discounting*PE/sims*100, label = 'EPE')
    sns.lineplot(x=time, y=discounting*NE/sims*100, label = 'ENE')
    plt.scatter(x=float_x, y=float_y, label = 'Float', marker = 'x', s = 60, c='black', linewidths=2)
    plt.scatter(x=fix_x, y=fix_y, label = 'Fix', marker = 'o', s = 60, facecolors='none', edgecolors='r', linewidths=2)
    fig.set_size_inches(15,8)
    # ax.set_ylim(-6,6)
    # ax.set_xlim(0,10.1)
    ax.set_xlabel('Time (Years)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('Exposure (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.axhline(y=0, color='k', alpha = 0.25)
    plt.grid(alpha = 0.25)
    plt.xticks(np.arange(0, 10.5, 0.5))
    plt.legend(frameon = False, fontsize = 18, loc='upper right')
    
    cva   = CVA(time, dt, PE/sims)*1000000
    dva   = DVA(time, dt, NE/sims)*1000000
    fca   = FCA(time, PE/sims, sp)*1000000
    fba   = FBA(time, NE/sims, sm)*1000000

    print(f'CVA: {cva:.2f}, DVA: {dva:.2f}, FCA: {fca:.2f}, FBA: {fba:.2f}')
    print(f'Total {cva+dva+fca+fba}')
    with open('SimulationTimesFVA.txt', 'a') as f:
        f.write(f'\n{sims},{int(1/dt)},{cva},{dva},{fca},{fba},{sp},{sm},10Y Payer Swap Exposure,{timer.time()-start:.2f}')

    
if False:
    print('5Y10Y Payer Swaption Exposure')
    start = timer.time()
    
    #Constructing time grid
    time = np.arange(0,10+dt,dt)
    time = time[np.where(time<=10)]

    #Constructing lagged grid
    if dt == 1/365:
        lagged_time = time[0:len(time)-2]
        ttso = time   #times to simulate on
    else:
        lagged_time = (time-lag)[1::]
        ttso = np.insert(time, np.arange(1,len(time)), lagged_time)

    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]

    def worker(i):
        float = HW.create_path(dt, 15, 0, i)[1]
        swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
        if swap[np.where(time==5)] < 0:
            swap[np.where(time>=5)]=0
        dfloat = np.where(swap>=0, float+sp, float+sm)
        D = np.cumprod(np.insert(np.exp(-dfloat[:-1:]*dt),0,1))
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return D*np.maximum(swap,0), D*np.minimum(swap,0)

    print('Starting parallel processing with {} cores'.format(nCPU))
    results = Parallel(n_jobs=nCPU)(delayed(worker)(i) for i in range(sims))
    # results = [worker(i) for i in range(sims)]
    print('Finished parallel processing')
    PE = np.zeros(len(time))
    NE = np.zeros(len(time))
    print('Starting sequential processing')
    for i in range(len(results)):
        PE += results[i][0]
        NE += results[i][1]
    print(f'Finished sequential processing in {timer.time()-start:.2f}s')
    print('Creating graph')

    # discounting = np.array([HW.ZCB(0,t) for t in time])
    discounting = 1
    sigmaP = 0
    sigmaM = 0
    for H in results:
        sigmaP += (discounting*(H[0]-PE/sims))**2
        sigmaM += (discounting*(H[1]-NE/sims))**2
    sigmaP = np.sqrt(sigmaP/(sims-1))
    sigmaM = np.sqrt(sigmaM/(sims-1))

    HPUB = discounting*PE/sims+3*sigmaP/np.sqrt(sims)
    HPLB = discounting*PE/sims-3*sigmaP/np.sqrt(sims)
    HMUB = discounting*NE/sims+3*sigmaM/np.sqrt(sims)
    HMLB = discounting*NE/sims-3*sigmaM/np.sqrt(sims)
    # sigmaP = 0
    # sigmaM = 0
    # EPE =  load("./SimulationData/PE_10Y_Swap_N=100000_dt=365.joblib")
    # ENE =  load("./SimulationData/NE_10Y_Swap_N=100000_dt=365.joblib")
    # for H in results:
    #     sigmaP += (discounting*(H[0]-EPE))**2
    #     sigmaM += (discounting*(H[1]-ENE))**2
    # sigmaP = np.sqrt(sigmaP/(sims-1))
    # sigmaM = np.sqrt(sigmaM/(sims-1))

    # HPUB = discounting*EPE+3*sigmaP/np.sqrt(sims)
    # HPLB = discounting*EPE-3*sigmaP/np.sqrt(sims)
    # HMUB = discounting*ENE+3*sigmaM/np.sqrt(sims)
    # HMLB = discounting*ENE-3*sigmaM/np.sqrt(sims)

    plt.rc('font',family='Times New Roman')
    float_x = np.arange(0.5,10.5,0.5)
    float_y = np.full((len(float_x),), -5.6)
    fix_x = np.arange(1,11)
    fix_y = np.full(len(fix_x), -5)

    fig, ax = plt.subplots()
    sns.lineplot(x=time, y=discounting*PE/sims*100, label = 'EPE')
    sns.lineplot(x=time, y=discounting*NE/sims*100, label = 'ENE')
    plt.scatter(x=float_x, y=float_y, label = 'Float', marker = 'x', s = 60, c='black', linewidths=2)
    plt.scatter(x=fix_x, y=fix_y, label = 'Fix', marker = 'o', s = 60, facecolors='none', edgecolors='r', linewidths=2)
    fig.set_size_inches(15,8)
    # ax.set_ylim(-6,6)
    # ax.set_xlim(0,10.1)
    ax.set_xlabel('Time (Years)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('Exposure (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.axhline(y=0, color='k', alpha = 0.25)
    plt.grid(alpha = 0.25)
    plt.xticks(np.arange(0, 10.5, 0.5))
    plt.legend(frameon = False, fontsize = 18, loc='upper right')
    
    cva   = CVA(time, dt, PE/sims)*1000000
    dva   = DVA(time, dt, NE/sims)*1000000
    fca   = FCA(time, PE/sims, sp)*1000000
    fba   = FBA(time, NE/sims, sm)*1000000

    print(f'CVA: {cva:.2f}, DVA: {dva:.2f}, FCA: {fca:.2f}, FBA: {fba:.2f}')
    print(f'Total {cva+dva+fca+fba}')
    with open('SimulationTimesFVA.txt', 'a') as f:
        f.write(f'\n{sims},{int(1/dt)},{cva},{dva},{fca},{fba},{sp},{sm},5Y10Y Payer Swaption Exposure,{timer.time()-start:.2f}')
