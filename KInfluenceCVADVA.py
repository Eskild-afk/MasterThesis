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
import pandas as pd

beta = np.array([1.879892, 0.579211, 	3.993992, 1.445091])
tau  = np.array([ 16.633491, 	0.319680])
reversion=0.08670264780833303 #0.13949636660880768
volatility=0.013928489964789946 #0.017793899652989272
HW = HullWhite(initial=0.02459103, reversion=reversion, volatility=volatility, b=beta, tau=tau)

nCPU=int(cpu_count()/1-0)
#Setting up tenor
T=np.arange(0,10+0.5,0.5)
S=np.arange(0,11,1)

# Other settings
dt   = 1/12
sims = 100
total_time = timer.time()

KVM=0 # Threshold for VM
MTA=0 # Minimum Transfer Amount
lag=2/365 # Lookback lag

Kvals = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
MTAvals = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])


if False:
    print('10Y Payer Swap K changes CVA and DVA')
    start = timer.time()
    
    #Constructing time grid
    time = np.arange(0,10+dt,dt)
    time = time[np.where(time <= 10)]
    #Constructing lagged grid
    
    if dt == 1/365:
        lagged_time = time[0:len(time)-2]
        ttso = time   #times to simulate on
    else:
        lagged_time = (time-lag)[1::]
        ttso = np.insert(time, np.arange(1,len(time)), lagged_time)
    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]

    def worker(i):
        float = [HW.init]
        VM = []
        ts=1
        for j in range(1, len(ttso)):
            ss=ttso[j]-ttso[j-1] #stepsize
            float.append(HW.oneStep(t=ttso[j], stepfrom=float[j-1], stepsize=ss, fwd=0))
        
        if dt==1/365:
            lagged_float = np.array(float[0:len(float)-2])
            float = np.array(float)
            ts+=2

            swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
            lagged_swap = swap
        else:
            lagged_float = np.array(float[1::2])
            float = np.array(float[0::2])

            swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
            lagged_swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=lagged_float, schedule=lagged_time, initRate=x[1]) for x in np.array([lagged_time,lagged_float]).T])

        for t in enumerate(time):
            if t[1]-lag<=0:
                VM.append(0)
                continue
            if t[1]>=time[-1]:
                VM.append(0)
                break
            VMhat = VM[t[0]-ts]/HW.ZCB(time[t[0]-1]-lag, time[t[0]]-lag, initRate=lagged_float[t[0]-ts]) # VM hat in eq 3.2 everything...
            VMta  = VMhat #(ta: to append) 
            
            VMta += (abs(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))>MTA)*(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))
            VMta += (abs(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))>MTA)*(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))

            VM.append(VMta)
            print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')

        D = np.cumprod(np.insert(np.exp(-float[:-1:]*dt),0,1))
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        exposure = np.where(swap >= 0,  np.maximum(swap-VM,0), np.minimum(swap-VM,0))
        return D*np.maximum(exposure,0), D*np.minimum(exposure,0)
        
    res = {
            'K':[],
            'CVA':[],
            'CVA upper bound':[],
            'CVA lower bound':[],
            'DVA':[],
            'DVA upper bound':[],
            'DVA lower bound':[]
        }

    for KVM in Kvals:
        print('Starting parallel processing with {} cores'.format(nCPU))
        results = Parallel(n_jobs=nCPU)(delayed(worker)(i) for i in range(sims))
        print('Finished parallel processing')
        PE = np.zeros(len(time))
        NE = np.zeros(len(time))
        print('Starting sequential processing')
        for i in range(len(results)):
            PE += results[i][0]
            NE += results[i][1]
        print(f'Finished sequential processing in {timer.time()-start:.2f} seconds')

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

        CVA_val = CVA(time, dt, PE)/sims
        CVA_UB = CVA(time, dt, HPUB)
        CVA_LB = CVA(time, dt, HPLB)

        DVA_val = DVA(time, dt, NE)/sims
        DVA_UB = DVA(time, dt, HMUB)
        DVA_LB = DVA(time, dt, HMLB)

        res['K'].append(KVM)
        res['CVA'].append(CVA_val)
        res['CVA upper bound'].append(CVA_UB)
        res['CVA lower bound'].append(CVA_LB)
        res['DVA'].append(DVA_val)
        res['DVA upper bound'].append(DVA_UB)
        res['DVA lower bound'].append(DVA_LB)

    results = pd.DataFrame(res)

    cvaval = [-5108.98]*len(results)
    cva_lower = [-5108.98*(1+0.061)]*len(results)
    cva_upper = [-5108.98*(1-0.061)]*len(results)
    #Upper three are all hardcoded from tabel with CVA/DVA with matching dt

    x = results['K']*100
    y = results['CVA']*1000000
    plt.rc('font',family='Times New Roman')
    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)
    sns.lineplot(x=x, y=y, label = 'CVA with VM')
    sns.lineplot(x=x, y=cvaval, label = 'CVA no collateral', c = 'r')
    sns.lineplot(x=x, y=cva_upper, linestyle = (0, (5,10)), c = 'r')
    sns.lineplot(x=x, y=cva_lower, linestyle = (0, (5,10)), c = 'r')
    ax.set_xlabel('K (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('CVA (EUR)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.fill_between(x, (results['CVA lower bound']*1000000), (results['CVA upper bound']*1000000), color='grey', alpha=0.1)
    plt.legend(frameon = False, fontsize = 18, loc='upper right')

    plt.savefig(f'./Graphs/K_Change_CVA_Swap_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')

    dvaval = [4599.89]*len(results)
    dva_lower = [4599.89*(1+0.061)]*len(results)
    dva_upper = [4599.89*(1-0.061)]*len(results)

    x = results['K']*100
    y = results['DVA']*1000000
    plt.rc('font',family='Times New Roman')
    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)
    sns.lineplot(x=x, y=y, label = 'DVA with VM')
    sns.lineplot(x=x, y=dvaval, label = 'DVA no collateral', c = 'r')
    sns.lineplot(x=x, y=dva_upper, linestyle = (0, (5,10)), c = 'r')
    sns.lineplot(x=x, y=dva_lower, linestyle = (0, (5,10)), c = 'r')
    ax.set_xlabel('K (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('DVA (EUR)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.fill_between(x, (results['DVA lower bound']*1000000), (results['DVA upper bound']*1000000), color='grey', alpha=0.1)
    plt.legend(frameon = False, fontsize = 18, loc='upper right')

    plt.savefig(f'./Graphs/K_Change_DVA_Swap_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')


if False:
    print('10Y Payer Swap Mta change CVA DVA')
    start = timer.time()
    
    #Constructing time grid
    time = np.arange(0,10+dt,dt)
    time = time[np.where(time <= 10)]
    #Constructing lagged grid
    
    if dt == 1/365:
        lagged_time = time[0:len(time)-2]
        ttso = time   #times to simulate on
    else:
        lagged_time = (time-lag)[1::]
        ttso = np.insert(time, np.arange(1,len(time)), lagged_time)
    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]

    def worker(i):
        float = [HW.init]
        VM = []
        ts=1
        for j in range(1, len(ttso)):
            ss=ttso[j]-ttso[j-1] #stepsize
            float.append(HW.oneStep(t=ttso[j], stepfrom=float[j-1], stepsize=ss, fwd=0))
        
        if dt==1/365:
            lagged_float = np.array(float[0:len(float)-2])
            float = np.array(float)
            ts+=2

            swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
            lagged_swap = swap
        else:
            lagged_float = np.array(float[1::2])
            float = np.array(float[0::2])

            swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
            lagged_swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=lagged_float, schedule=lagged_time, initRate=x[1]) for x in np.array([lagged_time,lagged_float]).T])

        for t in enumerate(time):
            if t[1]-lag<=0:
                VM.append(0)
                continue
            if t[1]>=time[-1]:
                VM.append(0)
                break
            VMhat = VM[t[0]-ts]/HW.ZCB(time[t[0]-1]-lag, time[t[0]]-lag, initRate=lagged_float[t[0]-ts]) # VM hat in eq 3.2 everything...
            VMta  = VMhat #(ta: to append) 
            
            VMta += (abs(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))>MTA)*(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))
            VMta += (abs(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))>MTA)*(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))

            VM.append(VMta)
            print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')

        D = np.cumprod(np.insert(np.exp(-float[:-1:]*dt),0,1))
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        exposure = np.where(swap >= 0,  np.maximum(swap-VM,0), np.minimum(swap-VM,0))
        return D*np.maximum(exposure,0), D*np.minimum(exposure,0)
        
    res = {
            'MTA':[],
            'CVA':[],
            'CVA upper bound':[],
            'CVA lower bound':[],
            'DVA':[],
            'DVA upper bound':[],
            'DVA lower bound':[]
        }

    for MTA in MTAvals:
        print('Starting parallel processing with {} cores'.format(nCPU))
        results = Parallel(n_jobs=nCPU)(delayed(worker)(i) for i in range(sims))
        print('Finished parallel processing')
        PE = np.zeros(len(time))
        NE = np.zeros(len(time))
        print('Starting sequential processing')
        for i in range(len(results)):
            PE += results[i][0]
            NE += results[i][1]
        print(f'Finished sequential processing in {timer.time()-start:.2f} seconds')

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

        CVA_val = CVA(time, dt, PE)/sims
        CVA_UB = CVA(time, dt, HPUB)
        CVA_LB = CVA(time, dt, HPLB)

        DVA_val = DVA(time, dt, NE)/sims
        DVA_UB = DVA(time, dt, HMUB)
        DVA_LB = DVA(time, dt, HMLB)

        res['MTA'].append(MTA)
        res['CVA'].append(CVA_val)
        res['CVA upper bound'].append(CVA_UB)
        res['CVA lower bound'].append(CVA_LB)
        res['DVA'].append(DVA_val)
        res['DVA upper bound'].append(DVA_UB)
        res['DVA lower bound'].append(DVA_LB)

    results = pd.DataFrame(res)

    cvaval = [-5108.98]*len(results)
    cva_lower = [-5108.98*(1+0.061)]*len(results)
    cva_upper = [-5108.98*(1-0.061)]*len(results)
    #Upper three are all hardcoded from tabel with CVA/DVA 

    x = results['MTA']*100
    y = results['CVA']*1000000
    plt.rc('font',family='Times New Roman')
    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)
    sns.lineplot(x=x, y=y, label = 'CVA with VM')
    sns.lineplot(x=x, y=cvaval, label = 'CVA no collateral', c = 'r')
    sns.lineplot(x=x, y=cva_upper, linestyle = (0, (5,10)), c = 'r')
    sns.lineplot(x=x, y=cva_lower, linestyle = (0, (5,10)), c = 'r')
    ax.set_xlabel('MTA (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('CVA (EUR)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.fill_between(x, (results['CVA lower bound']*1000000), (results['CVA upper bound']*1000000), color='grey', alpha=0.1)
    plt.legend(frameon = False, fontsize = 18, loc='upper right')

    plt.savefig(f'./Graphs/MTA_Change_CVA_Swap_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')

    dvaval = [4599.89]*len(results)
    dva_lower = [4599.89*(1+0.061)]*len(results)
    dva_upper = [4599.89*(1-0.061)]*len(results)

    x = results['MTA']*100
    y = results['DVA']*1000000
    plt.rc('font',family='Times New Roman')
    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)
    sns.lineplot(x=x, y=y, label = 'DVA with VM')
    sns.lineplot(x=x, y=dvaval, label = 'DVA no collateral', c = 'r')
    sns.lineplot(x=x, y=dva_upper, linestyle = (0, (5,10)), c = 'r')
    sns.lineplot(x=x, y=dva_lower, linestyle = (0, (5,10)), c = 'r')
    ax.set_xlabel('MTA (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('DVA (EUR)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.fill_between(x, (results['DVA lower bound']*1000000), (results['DVA upper bound']*1000000), color='grey', alpha=0.1)
    plt.legend(frameon = False, fontsize = 18, loc='upper right')

    plt.savefig(f'./Graphs/MTA_Change_DVA_Swap_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')

if False:
    print('5x10Y Payer Swaption K changes CVA and DVA')
    start = timer.time()
    
    #Constructing time grid
    time = np.arange(0,10+5+dt,dt)
    time = time[np.where(time <= 15)]
    #Constructing lagged grid
    if dt == 1/365:
        lagged_time = time[0:len(time)-2]
        ttso = time   #times to simulate on
    else:
        lagged_time = (time-lag)[1::]
        ttso = np.insert(time, np.arange(1,len(time)), lagged_time)

    K=fsolve(lambda x: HW.swap(0, S+5, T+5, x), x0=0.02)[0]

    def worker(i):
        float = [HW.init]
        VM = []
        ts = 1
        for j in range(1, len(ttso)):
            ss=ttso[j]-ttso[j-1] #stepsize
            float.append(HW.oneStep(t=ttso[j], stepfrom=float[j-1], stepsize=ss, fwd=0))
        
        if dt==1/365:
            lagged_float = np.array(float[0:len(float)-2])
            float = np.array(float)
            ts += 2

            swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
            lagged_swap = swap
        else:
            lagged_float = np.array(float[1::2])
            float = np.array(float[0::2])

            swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
            lagged_swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=lagged_float, schedule=lagged_time) for x in np.array([lagged_time,lagged_float]).T])

        
        if swap[np.where(time==5)] < 0:
            swap[np.where(time>=5)]=0
            lagged_swap[np.where(lagged_time>=5)]=0

        for t in enumerate(time):
            if t[1]-lag <= 0:
                VM.append(0)
                continue
            if t[1] >= time[-1]:
                VM.append(0)
                break

            VMhat = VM[t[0]-ts]/HW.ZCB(time[t[0]-1]-lag, time[t[0]]-lag, initRate=lagged_float[t[0]-ts]) # VM hat in eq 3.2 everything...
            VMta  = VMhat #(ta: to append) 
            
            VMta += (abs(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))>MTA)*(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))
            VMta += (abs(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))>MTA)*(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))

            VM.append(VMta)
            print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')

        D = np.cumprod(np.insert(np.exp(-float[:-1:]*dt),0,1))
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        exposure = np.where(swap >= 0,  np.maximum(swap-VM,0), np.minimum(swap-VM,0))
        return D*np.maximum(exposure,0), D*np.minimum(exposure,0)
        
    res = {
            'K':[],
            'CVA':[],
            'CVA upper bound':[],
            'CVA lower bound':[],
            'DVA':[],
            'DVA upper bound':[],
            'DVA lower bound':[]
        }

    for KVM in Kvals:
        print('Starting parallel processing with {} cores'.format(nCPU))
        results = Parallel(n_jobs=nCPU)(delayed(worker)(i) for i in range(sims))
        print('Finished parallel processing')
        PE = np.zeros(len(time))
        NE = np.zeros(len(time))
        print('Starting sequential processing')
        for i in range(len(results)):
            PE += results[i][0]
            NE += results[i][1]
        print(f'Finished sequential processing in {timer.time()-start:.2f} seconds')

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

        CVA_val = CVA(time, dt, PE)/sims
        CVA_UB = CVA(time, dt, HPUB)
        CVA_LB = CVA(time, dt, HPLB)

        DVA_val = DVA(time, dt, NE)/sims
        DVA_UB = DVA(time, dt, HMUB)
        DVA_LB = DVA(time, dt, HMLB)

        res['K'].append(KVM)
        res['CVA'].append(CVA_val)
        res['CVA upper bound'].append(CVA_UB)
        res['CVA lower bound'].append(CVA_LB)
        res['DVA'].append(DVA_val)
        res['DVA upper bound'].append(DVA_UB)
        res['DVA lower bound'].append(DVA_LB)

    results = pd.DataFrame(res)

    cvaval = [-9678.3]*len(results)
    cva_lower = [-9678.3*(1+0.046)]*len(results)
    cva_upper = [-9678.3*(1-0.046)]*len(results)
    #Upper three are all hardcoded from tabel with CVA/DVA with matching dt

    x = results['K']*100
    y = results['CVA']*1000000
    plt.rc('font',family='Times New Roman')
    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)
    sns.lineplot(x=x, y=y, label = 'CVA with VM')
    sns.lineplot(x=x, y=cvaval, label = 'CVA no collateral', c = 'r')
    sns.lineplot(x=x, y=cva_upper, linestyle = (0, (5,10)), c = 'r')
    sns.lineplot(x=x, y=cva_lower, linestyle = (0, (5,10)), c = 'r')
    ax.set_xlabel('K (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('CVA (EUR)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.fill_between(x, (results['CVA lower bound']*1000000), (results['CVA upper bound']*1000000), color='grey', alpha=0.1)
    plt.legend(frameon = False, fontsize = 18, loc='upper right')

    plt.savefig(f'./Graphs/K_Change_CVA_Swaption_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')

    dvaval = [753.44]*len(results)
    dva_lower = [753.44*(1+0.145)]*len(results)
    dva_upper = [753.44*(1-0.145)]*len(results)

    x = results['K']*100
    y = results['DVA']*1000000
    plt.rc('font',family='Times New Roman')
    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)
    sns.lineplot(x=x, y=y, label = 'DVA with VM')
    sns.lineplot(x=x, y=dvaval, label = 'DVA no collateral', c = 'r')
    sns.lineplot(x=x, y=dva_upper, linestyle = (0, (5,10)), c = 'r')
    sns.lineplot(x=x, y=dva_lower, linestyle = (0, (5,10)), c = 'r')
    ax.set_xlabel('K (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('DVA (EUR)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.fill_between(x, (results['DVA lower bound']*1000000), (results['DVA upper bound']*1000000), color='grey', alpha=0.1)
    plt.legend(frameon = False, fontsize = 18, loc='upper right')

    plt.savefig(f'./Graphs/K_Change_DVA_Swaption_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')


if True:
    print('5x10Y Payer Swaption Mta change CVA DVA')
    start = timer.time()
    
    #Constructing time grid
    time = np.arange(0,10+5+dt,dt)
    time = time[np.where(time <= 15)]
    #Constructing lagged grid
    if dt == 1/365:
        lagged_time = time[0:len(time)-2]
        ttso = time   #times to simulate on
    else:
        lagged_time = (time-lag)[1::]
        ttso = np.insert(time, np.arange(1,len(time)), lagged_time)

    K=fsolve(lambda x: HW.swap(0, S+5, T+5, x), x0=0.02)[0]

    def worker(i):
        float = [HW.init]
        VM = []
        ts = 1
        for j in range(1, len(ttso)):
            ss=ttso[j]-ttso[j-1] #stepsize
            float.append(HW.oneStep(t=ttso[j], stepfrom=float[j-1], stepsize=ss, fwd=0))
        
        if dt==1/365:
            lagged_float = np.array(float[0:len(float)-2])
            float = np.array(float)
            ts += 2

            swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
            lagged_swap = swap
        else:
            lagged_float = np.array(float[1::2])
            float = np.array(float[0::2])

            swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
            lagged_swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=lagged_float, schedule=lagged_time) for x in np.array([lagged_time,lagged_float]).T])

        
        if swap[np.where(time==5)] < 0:
            swap[np.where(time>=5)]=0
            lagged_swap[np.where(lagged_time>=5)]=0

        for t in enumerate(time):
            if t[1]-lag <= 0:
                VM.append(0)
                continue
            if t[1] >= time[-1]:
                VM.append(0)
                break

            VMhat = VM[t[0]-ts]/HW.ZCB(time[t[0]-1]-lag, time[t[0]]-lag, initRate=lagged_float[t[0]-ts]) # VM hat in eq 3.2 everything...
            VMta  = VMhat #(ta: to append) 
            
            VMta += (abs(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))>MTA)*(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))
            VMta += (abs(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))>MTA)*(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))

            VM.append(VMta)
            print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')

        D = np.cumprod(np.insert(np.exp(-float[:-1:]*dt),0,1))
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        exposure = np.where(swap >= 0,  np.maximum(swap-VM,0), np.minimum(swap-VM,0))
        return D*np.maximum(exposure,0), D*np.minimum(exposure,0)
        
    res = {
            'MTA':[],
            'CVA':[],
            'CVA upper bound':[],
            'CVA lower bound':[],
            'DVA':[],
            'DVA upper bound':[],
            'DVA lower bound':[]
        }

    for MTA in MTAvals:
        print('Starting parallel processing with {} cores'.format(nCPU))
        results = Parallel(n_jobs=nCPU)(delayed(worker)(i) for i in range(sims))
        print('Finished parallel processing')
        PE = np.zeros(len(time))
        NE = np.zeros(len(time))
        print('Starting sequential processing')
        for i in range(len(results)):
            PE += results[i][0]
            NE += results[i][1]
        print(f'Finished sequential processing in {timer.time()-start:.2f} seconds')

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

        CVA_val = CVA(time, dt, PE)/sims
        CVA_UB = CVA(time, dt, HPUB)
        CVA_LB = CVA(time, dt, HPLB)

        DVA_val = DVA(time, dt, NE)/sims
        DVA_UB = DVA(time, dt, HMUB)
        DVA_LB = DVA(time, dt, HMLB)

        res['MTA'].append(MTA)
        res['CVA'].append(CVA_val)
        res['CVA upper bound'].append(CVA_UB)
        res['CVA lower bound'].append(CVA_LB)
        res['DVA'].append(DVA_val)
        res['DVA upper bound'].append(DVA_UB)
        res['DVA lower bound'].append(DVA_LB)

    results = pd.DataFrame(res)

    cvaval = [-9678.3]*len(results)
    cva_lower = [-9678.3*(1+0.046)]*len(results)
    cva_upper = [-9678.3*(1-0.046)]*len(results)
    #Upper three are all hardcoded from tabel with CVA/DVA 

    x = results['MTA']*100
    y = results['CVA']*1000000
    plt.rc('font',family='Times New Roman')
    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)
    sns.lineplot(x=x, y=y, label = 'CVA with VM')
    sns.lineplot(x=x, y=cvaval, label = 'CVA no collateral', c = 'r')
    sns.lineplot(x=x, y=cva_upper, linestyle = (0, (5,10)), c = 'r')
    sns.lineplot(x=x, y=cva_lower, linestyle = (0, (5,10)), c = 'r')
    ax.set_xlabel('MTA (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('CVA (EUR)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.fill_between(x, (results['CVA lower bound']*1000000), (results['CVA upper bound']*1000000), color='grey', alpha=0.1)
    plt.legend(frameon = False, fontsize = 18, loc='upper right')

    plt.savefig(f'./Graphs/MTA_Change_CVA_Swap_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')

    dvaval = [753.44]*len(results)
    dva_lower = [753.44*(1+0.145)]*len(results)
    dva_upper = [753.44*(1-0.145)]*len(results)

    x = results['MTA']*100
    y = results['DVA']*1000000
    plt.rc('font',family='Times New Roman')
    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)
    sns.lineplot(x=x, y=y, label = 'DVA with VM')
    sns.lineplot(x=x, y=dvaval, label = 'DVA no collateral', c = 'r')
    sns.lineplot(x=x, y=dva_upper, linestyle = (0, (5,10)), c = 'r')
    sns.lineplot(x=x, y=dva_lower, linestyle = (0, (5,10)), c = 'r')
    ax.set_xlabel('MTA (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('DVA (EUR)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.fill_between(x, (results['DVA lower bound']*1000000), (results['DVA upper bound']*1000000), color='grey', alpha=0.1)
    plt.legend(frameon = False, fontsize = 18, loc='upper right')

    plt.savefig(f'./Graphs/MTA_Change_DVA_Swaption_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')