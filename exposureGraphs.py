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
        D = np.cumprod(np.insert(np.exp(-float[:-1:]*dt),0,1))
        swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
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
    dump(PE/sims, f'./SimulationData/PE_10Y_Swap_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_10Y_Swap_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_10YPayer_Swap_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    
    cva   = CVA(time, dt, discounting*PE/sims)
    cvaUB = CVA(time, dt, HPUB)
    cvaLB = CVA(time, dt, HPLB)
    dva   = DVA(time, dt, discounting*NE/sims)
    dvaUB = DVA(time, dt, HMUB)
    dvaLB = DVA(time, dt, HMLB)


    print( 'Finished creating graph')
    with open('SimulationTimes.txt', 'a') as f:
        f.write(f'\n{sims},{int(1/dt)},{cva},{cvaUB},{cvaLB},{dva},{dvaUB},{dvaLB},10Y Payer Swap Exposure,{timer.time()-start:.2f}')
sys.exit()
# 5Y10YForward Swap
if False:
    print('5Y10Y Forward Payer Swap Exposure')
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
        for j in range(1, len(ttso)):
            ss=ttso[j]-ttso[j-1] #stepsize
            float.append(HW.oneStep(t=ttso[j], stepfrom=float[j-1], stepsize=ss, fwd=0))
        
        if dt==1/365:
            lagged_float = np.array(float[0:len(float)-2])
            float = np.array(float)
        else:
            lagged_float = np.array(float[1::2])
            float = np.array(float[0::2])

        swap = np.array([HW.swapextended(x[0], S+5, T+5, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return np.maximum(swap,0), np.minimum(swap,0)


    # def worker(i):
    #     time, float = HW.create_path(dt,15, seed=i)
    #     swap = np.array([HW.swapextended(x[0], S+5, T+5, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
    #     print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
    #     return np.maximum(swap,0), np.minimum(swap,0)

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
    print('Creating graph')
    discounting = np.array([HW.marketZCB(t) for t in time])
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
    plt.rc('font',family='Times New Roman')
    float_x = np.arange(5.5,15.5,0.5)
    float_y = np.full((len(float_x),), -6.6)
    fix_x = np.arange(6,16)
    fix_y = np.full(len(fix_x), -6)

    fig, ax = plt.subplots()
    sns.lineplot(x=time, y=discounting*PE/sims*100, label = 'EPE')
    sns.lineplot(x=time, y=discounting*NE/sims*100, label = 'ENE')
    plt.scatter(x=float_x, y=float_y, label = 'Float', marker = 'x', s = 60, c='black', linewidths=2)
    plt.scatter(x=fix_x, y=fix_y, label = 'Fix', marker = 'o', s = 60, facecolors='none', edgecolors='r', linewidths=2)
    fig.set_size_inches(15,8)
    ax.set_ylim(-7,6)
    ax.set_xlim(0,15.1)
    ax.set_xlabel('Time (Years)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('Exposure (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.axhline(y=0, color='k', alpha = 0.25)
    plt.grid(alpha = 0.25)
    plt.xticks(np.arange(0, 16))
    plt.legend(frameon = False, fontsize = 18, loc='upper right')
    dump(PE/sims, f'./SimulationData/PE_5Y10Y_ForwardSwap_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_5Y10Y_ForwardSwap_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_5Y10YForwardPayer_Swap_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')
    cva   = CVA(time, dt, discounting*PE/sims)
    cvaUB = CVA(time, dt, HPUB)
    cvaLB = CVA(time, dt, HPLB)
    dva   = DVA(time, dt, discounting*NE/sims)
    dvaUB = DVA(time, dt, HMUB)
    dvaLB = DVA(time, dt, HMLB)
    with open('SimulationTimes.txt', 'a') as f:
        f.write(f'\n{sims},{int(1/dt)},{cva},{cvaUB},{cvaLB},{dva},{dvaUB},{dvaLB},5Y10YForward Swap Exposure,{timer.time()-start:.2f}')
#5Y10Y Payer Swaption Exposure
if False:
    print('5Y10Y Payer Swaption Exposure')
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
        float = HW.create_path(dt, 15, 0, i)[1]
        swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
        D = np.cumprod(np.insert(np.exp(-float[:-1:]*dt),0,1))
        if swap[np.where(time==5)] < 0:
            swap[np.where(time>=5)]=0
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return D*np.maximum(swap,0), D*np.minimum(swap,0)

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
    print('Creating graph')
    discounting = 1 #np.array([HW.marketZCB(t) for t in time])
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
    plt.rc('font',family='Times New Roman')
    float_x = np.arange(5.5,15.5,0.5)
    float_y = np.full((len(float_x),), -2.6)
    fix_x = np.arange(6,16)
    fix_y = np.full(len(fix_x), -2)

    fig, ax = plt.subplots()
    sns.lineplot(x=time, y=discounting*PE/sims*100, label = 'EPE')
    sns.lineplot(x=time, y=discounting*NE/sims*100, label = 'ENE')
    plt.scatter(x=float_x, y=float_y, label = 'Float', marker = 'x', s = 60, c='black', linewidths=2)
    plt.scatter(x=fix_x, y=fix_y, label = 'Fix', marker = 'o', s = 60, facecolors='none', edgecolors='r', linewidths=2)
    fig.set_size_inches(15,8)
    # ax.set_ylim(-6,6)time, float = HW.create_path(1/365,10, seed=0)
    ax.set_xlim(0,15.1)
    ax.set_xlabel('Time (Years)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('Exposure (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.axhline(y=0, color='k', alpha = 0.25)
    plt.grid(alpha = 0.25)
    plt.xticks(np.arange(0, 16))
    plt.legend(frameon = False, fontsize = 18, loc='upper right')
    dump(PE/sims, f'./SimulationData/PE_5Y10YPayer_Swaption_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_5Y10YPayer_Swaption_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_5Y10YPayer_Swaption_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')
    cva   = CVA(time, dt, discounting*PE/sims)
    cvaUB = CVA(time, dt, HPUB)
    cvaLB = CVA(time, dt, HPLB)
    dva   = DVA(time, dt, discounting*NE/sims)
    dvaUB = DVA(time, dt, HMUB)
    dvaLB = DVA(time, dt, HMLB)
    with open('SimulationTimes.txt', 'a') as f:
        f.write(f'\n{sims},{int(1/dt)},{cva},{cvaUB},{cvaLB},{dva},{dvaUB},{dvaLB},5Y10Y Payer Swaption Exposure,{timer.time()-start:.2f}')
# Swap with Variation Margin

if False:
    print('10Y Payer Swap with Variation Margin')
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
    print('Creating graph')
    discounting = 1#np.array([HW.marketZCB(t) for t in time])
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
    plt.rc('font',family='Times New Roman')
    float_x = np.arange(0.5,10.5,0.5)
    float_y = np.full((len(float_x),), -5.6)
    fix_x = np.arange(1,11)
    fix_y = np.full(len(fix_x), -5)

    fig, ax = plt.subplots()
    sns.lineplot(x=time, y=discounting*PE/sims*100, label = 'EPE')
    sns.lineplot(x=time, y=discounting*NE/sims*100, label = 'ENE')
    # plt.scatter(x=float_x, y=float_y, label = 'Float', marker = 'x', s = 60, c='black', linewidths=2)
    # plt.scatter(x=fix_x, y=fix_y, label = 'Fix', marker = 'o', s = 60, facecolors='none', edgecolors='r', linewidths=2)
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
    dump(PE/sims, f'./SimulationData/PE_10Y_Swap_With_VM_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_10Y_Swap_With_VM_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_10YPayer_Swap_With_VM_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')
    cva   = CVA(time, dt, discounting*PE/sims)
    cvaUB = CVA(time, dt, HPUB)
    cvaLB = CVA(time, dt, HPLB)
    dva   = DVA(time, dt, discounting*NE/sims)
    dvaUB = DVA(time, dt, HMUB)
    dvaLB = DVA(time, dt, HMLB)
    with open('SimulationTimes.txt', 'a') as f:
        f.write(f'\n{sims},{int(1/dt)},{cva},{cvaUB},{cvaLB},{dva},{dvaUB},{dvaLB},10Y Payer Swap with VM,{timer.time()-start:.2f}')      

# Swaption with Variation Margin
if False:
    print('5Y10Y Payer Swaption Exposure with Variation Margin')
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
    print('Creating graph')
    discounting = 1#np.array([HW.marketZCB(t) for t in time])
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
    plt.rc('font',family='Times New Roman')
    float_x = np.arange(5.5,15.5,0.5)
    float_y = np.full((len(float_x),), -2.6)
    fix_x = np.arange(6,16)
    fix_y = np.full(len(fix_x), -2)

    fig, ax = plt.subplots()
    sns.lineplot(x=time[2::], y=(PE/sims*100)[2::], label = 'EPE')
    sns.lineplot(x=time[2::], y=(NE/sims*100)[2::], label = 'ENE')
    plt.scatter(x=float_x, y=float_y, label = 'Float', marker = 'x', s = 60, c='black', linewidths=2)
    plt.scatter(x=fix_x, y=fix_y, label = 'Fix', marker = 'o', s = 60, facecolors='none', edgecolors='r', linewidths=2)
    fig.set_size_inches(15,8)
    # ax.set_ylim(-6,6)time, float = HW.create_path(1/365,10, seed=0)
    ax.set_xlim(0,15.1)
    ax.set_xlabel('Time (Years)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('Exposure (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.axhline(y=0, color='k', alpha = 0.25)
    plt.grid(alpha = 0.25)
    plt.xticks(np.arange(0, 16))
    plt.legend(frameon = False, fontsize = 18, loc='upper right')
    dump(PE/sims, f'./SimulationData/PE_5Y10Y_Swaption_With_VM_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_5Y10Y_Swaption_With_VM_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_5Y10YPayer_Swaption_With_VM_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')
    cva   = CVA(time, dt, discounting*PE/sims)
    cvaUB = CVA(time, dt, HPUB)
    cvaLB = CVA(time, dt, HPLB)
    dva   = DVA(time, dt, discounting*NE/sims)
    dvaUB = DVA(time, dt, HMUB)
    dvaLB = DVA(time, dt, HMLB)
    with open('SimulationTimes.txt', 'a') as f:
        f.write(f'\n{sims},{int(1/dt)},{cva},{cvaUB},{cvaLB},{dva},{dvaUB},{dvaLB},5Y10YPayer Swaption with VM,{timer.time()-start:.2f}')

#Swap 10Y with VM and IM
if True:
    print('10Y Payer Swap with Variation and Initial Margin')
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

    oneBP=0.0001
    RW_6M=53
    T_b=230

    def worker(i):
        float = [HW.init]
        VM = []
        IM = []
        ts = 1
        for j in range(1, len(ttso)):
            ss=ttso[j]-ttso[j-1] #stepsize
            float.append(HW.oneStep(t=ttso[j], stepfrom=float[j-1], stepsize=ss, fwd=0))
        
        if dt==1/365:
            lagged_float = np.array(float[0:len(float)-2])
            float = np.array(float)
            ts+=2

            swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
            lagged_swap = swap
            swapBP = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=lagged_float+oneBP, schedule=lagged_time, initRate=x[1]+oneBP) for x in np.array([lagged_time,lagged_float]).T])

        else:
            lagged_float = np.array(float[1::2])
            float = np.array(float[0::2])

            swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
            lagged_swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=lagged_float, schedule=lagged_time, initRate=x[1]) for x in np.array([lagged_time,lagged_float]).T])
            swapBP = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=lagged_float+oneBP, schedule=lagged_time, initRate=x[1]+oneBP) for x in np.array([lagged_time,lagged_float]).T])
        
            
        for t in enumerate(time):
            if t[1]-lag <= 0:
                VM.append(0)
                IM.append(0)
                continue
            if t[1] >= time[-1]:
                VM.append(0)
                IM.append(0)
                break

            #VM calc
            VMhat = VM[t[0]-ts]/HW.ZCB(time[t[0]-1]-lag, time[t[0]]-lag, initRate=lagged_float[t[0]-ts]) # VM hat in eq 3.2 everything...
            VMta  = VMhat #(ta: to append) 
            
            VMta += (abs(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))>MTA)*(pos(lagged_swap[t[0]-ts]-KVM)-pos(VMhat))
            VMta += (abs(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))>MTA)*(neg(lagged_swap[t[0]-ts]+KVM)-neg(VMhat))
            
            VM.append(VMta)

            #IM calc
            s = swapBP[t[0]-ts]-lagged_swap[t[0]-ts]
            CR_B = np.maximum(1,np.sqrt(abs(s)/T_b))
            IMta = CR_B*s*RW_6M

            IM.append((np.maximum(IMta-KIM,0)>MTA)*np.maximum(IMta-KIM,0))
        D = np.cumprod(np.insert(np.exp(-float[:-1:]*dt),0,1)) 
        result = np.where(swap>0, np.maximum(swap-VM-IM,0), np.minimum(swap-VM+IM,0))
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return D*np.maximum(result, 0), D*np.minimum(result, 0)
        
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
    print('Creating graph')
    discounting = 1#np.array([HW.marketZCB(t) for t in time])
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
    plt.rc('font',family='Times New Roman')
    float_x = np.arange(0.5,10.5,0.5)
    float_y = np.full((len(float_x),), -5.6)
    fix_x = np.arange(1,11)
    fix_y = np.full(len(fix_x), -5)

    fig, ax = plt.subplots()
    sns.lineplot(x=time, y=discounting*PE/sims*100, label = 'EPE')
    sns.lineplot(x=time, y=discounting*NE/sims*100, label = 'ENE')
    # plt.scatter(x=float_x, y=float_y, label = 'Float', marker = 'x', s = 60, c='black', linewidths=2)
    # plt.scatter(x=fix_x, y=fix_y, label = 'Fix', marker = 'o', s = 60, facecolors='none', edgecolors='r', linewidths=2)
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
    dump(PE/sims, f'./SimulationData/PE_10Y_Swap_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_10Y_Swap_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_10YPayer_Swap_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')
    cva   = CVA(time, dt, discounting*PE/sims)
    cvaUB = CVA(time, dt, HPUB)
    cvaLB = CVA(time, dt, HPLB)
    dva   = DVA(time, dt, discounting*NE/sims)
    dvaUB = DVA(time, dt, HMUB)
    dvaLB = DVA(time, dt, HMLB)
    with open('SimulationTimes.txt', 'a') as f:
        f.write(f'\n{sims},{int(1/dt)},{cva},{cvaUB},{cvaLB},{dva},{dvaUB},{dvaLB},10Y Payer Swap with VM+IM,{timer.time()-start:.2f}')

#Swaption with VM and IM
if True:
    print('5Y10Y Payer Swaption with Variation and Initial Margin')
    start = timer.time()
    #Constructing time grid
    time = np.arange(0,15+dt,dt)
    time = time[np.where(time <= 15)]

    #Constructing lagged grid
    if dt == 1/365:
        lagged_time = time[0:len(time)-2]
        ttso = time   #times to simulate on
    else:
        lagged_time = (time-lag)[1::]
        ttso = np.insert(time, np.arange(1,len(time)), lagged_time)

    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]

    oneBP=0.0001
    RW_6M=53
    T_b=230

    def worker(i):
        float = [HW.init]
        IM = []
        VM = []
        ts=1

        for j in range(1, len(ttso)):
            ss=ttso[j]-ttso[j-1] #stepsize
            float.append(HW.oneStep(t=ttso[j], stepfrom=float[j-1], stepsize=ss, fwd=0))
        

        if dt==1/365:
            lagged_float = np.array(float[0:len(float)-2])
            float = np.array(float)
            ts+=2
            swaption = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])

            swaptionBP = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1]+oneBP, floatRate=lagged_float+oneBP, schedule=lagged_time,) for x in np.array([lagged_time,lagged_float]).T])
            lagged_swaption = swaption
            
        else:
            lagged_float = np.array(float[1::2])
            float = np.array(float[0::2])

            swaption = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
            swaptionBP = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1]+oneBP, floatRate=lagged_float+oneBP, schedule=lagged_time,) for x in np.array([lagged_time,lagged_float]).T])
            lagged_swaption = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=lagged_float, schedule=lagged_time) for x in np.array([lagged_time,lagged_float]).T])
            
        if swaption[np.where(time==5)] < 0:
            swaption[np.where(time>=5)]=0
            lagged_swaption[np.where(lagged_time>=5)]=0
            swaptionBP[np.where(lagged_time>=5)]=0

        for t in enumerate(time):
            if t[1]-lag <= 0:
                VM.append(0)
                IM.append(0)
                continue
            if t[1] >= time[-1]:
                VM.append(0)
                IM.append(0)
                break

            #VM calc
            VMhat = VM[t[0]-ts]/HW.ZCB(time[t[0]-1]-lag, time[t[0]]-lag, initRate=lagged_float[t[0]-ts]) # VM hat in eq 3.2 everything...
            VMta  = VMhat #(ta: to append) 
            
            VMta += (abs(pos(lagged_swaption[t[0]-ts]-KVM)-pos(VMhat))>MTA)*(pos(lagged_swaption[t[0]-ts]-KVM)-pos(VMhat))
            VMta += (abs(neg(lagged_swaption[t[0]-ts]+KVM)-neg(VMhat))>MTA)*(neg(lagged_swaption[t[0]-ts]+KVM)-neg(VMhat))

            VM.append(VMta)

            #IM calc
            s = swaptionBP[t[0]-ts]-lagged_swaption[t[0]-ts]
            CR_B = np.maximum(1,np.sqrt(abs(s)/T_b))
            IM.append(CR_B*s*RW_6M)
            #print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
                        
        exposure = np.where(swaption > 0, np.maximum(swaption-VM-IM,0), np.minimum(swaption-VM+IM,0))
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        D = np.cumprod(np.insert(np.exp(-float[:-1:]*dt),0,1))
        return D*np.maximum(exposure, 0), D*np.minimum(exposure, 0)
        
        

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
    print('Creating graph')
    discounting = 1#np.array([HW.marketZCB(t) for t in time])
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
    plt.rc('font',family='Times New Roman')
    #float_x = np.arange(5.5,15.5,0.5)
    #float_y = np.full((len(float_x),), -5.6)
    #fix_x = np.arange(1,11)
    #fix_y = np.full(len(fix_x), -5)

    fig, ax = plt.subplots()
    sns.lineplot(x=time, y=discounting*PE/sims*100, label = 'EPE')
    sns.lineplot(x=time, y=discounting*NE/sims*100, label = 'ENE')
    # plt.scatter(x=float_x, y=float_y, label = 'Float', marker = 'x', s = 60, c='black', linewidths=2)
    # plt.scatter(x=fix_x, y=fix_y, label = 'Fix', marker = 'o', s = 60, facecolors='none', edgecolors='r', linewidths=2)
    fig.set_size_inches(15,8)
    ax.set_ylim(-0.6,0.2)
    ax.set_xlim(0,15.1)
    ax.set_xlabel('Time (Years)', fontname="Times New Roman", fontsize = 28)
    ax.set_ylabel('Exposure (% Notional)', fontname="Times New Roman", fontsize = 28)
    ax.tick_params(axis='x', direction='in', right = 'True', labelsize = 24, pad = 15)
    ax.tick_params(axis='y', direction='in', top = 'True', labelsize = 24, pad = 15)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.axhline(y=0, color='k', alpha = 0.25)
    plt.grid(alpha = 0.25)
    plt.xticks(np.arange(0, 16, 5))
    plt.legend(frameon = False, fontsize = 18, loc='upper right')
    dump(PE/sims, f'./SimulationData/PE_5Y10Y_Swaption_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_5Y10Y_Swaption_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_5Y10YPayer_Swaption_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')
    cva   = CVA(time, dt, discounting*PE/sims)
    cvaUB = CVA(time, dt, HPUB)
    cvaLB = CVA(time, dt, HPLB)
    dva   = DVA(time, dt, discounting*NE/sims)
    dvaUB = DVA(time, dt, HMUB)
    dvaLB = DVA(time, dt, HMLB)
    with open('SimulationTimes.txt', 'a') as f:
        f.write(f'\n{sims},{int(1/dt)},{cva},{cvaUB},{cvaLB},{dva},{dvaUB},{dvaLB},10Y Payer Swaption with VM+IM,{timer.time()-start:.2f}')


print(f'Total Time: {timer.time()-total_time}')