import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, cpu_count, dump
from scipy.optimize import fsolve
from HullWhite import HullWhite
import time as timer

import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths='C:\Windows\Fonts', fontext='ttf')

beta = np.array([1.879892, 0.579211, 	3.993992, 1.445091])
tau  = np.array([ 16.633491, 	0.319680])
reversion=0.13949636660880768 
volatility=0.017793899652989272
# HW = HullWhite(initial=0.02459103, reversion=0.03, volatility=0.00200, Gamma=1000, b=beta, tau=tau)
HW = HullWhite(initial=0.02459103, reversion=reversion, volatility=volatility, Gamma=1000, b=beta, tau=tau)

dt = 1/365

#Setting up tenor
T=np.arange(0,10+0.5,0.5)
S=np.arange(0,11,1)

#Helper functions 
def pos (x):
    return np.maximum(x,0)

def neg (x):
    return np.minimum(x,0)

#10Y Payer Swap Exposure
sims = 10
if False:
    start = timer.time()
    time, float = HW.create_path(dt,10, seed=0)
    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]


    def worker(i):
        time, float = HW.create_path(dt,10, seed=i)
        swap = np.array([HW.swap(x[0], S, T, K=K, initRate=x[1]) for x in np.array([time,float]).T])
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return np.maximum(swap,0), np.minimum(swap,0)

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=cpu_count())(delayed(worker)(i) for i in range(sims))
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
    ax.set_ylim(-6,6)
    ax.set_xlim(0,10.1)
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
    
    print( 'Finished creating graph')
# 5Y10YForward Swap
if True:
    start = timer.time()
    time, float = HW.create_path(dt,15, seed=0)
    K=fsolve(lambda x: HW.swap(0, S+5, T+5, x), x0=0.02)[0]


    def worker(i):
        time, float = HW.create_path(dt,15, seed=i)
        swap = np.array([HW.swap(x[0], S+5, T+5, K=K, initRate=x[1]) for x in np.array([time,float]).T])
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return np.maximum(swap,0), np.minimum(swap,0)

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=cpu_count())(delayed(worker)(i) for i in range(sims))
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

    plt.rc('font',family='Times New Roman')
    float_x = np.arange(0.5,15.5,0.5)
    float_y = np.full((len(float_x),), -6.6)
    fix_x = np.arange(1,16)
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

#5Y10Y Payer Swaption Exposure
if True:
    start = timer.time()
    time, float = HW.create_path(dt,15, seed=0)
    from joblib import Parallel, delayed, cpu_count
    K=fsolve(lambda x: HW.swap(0, S+5, T+5, x), x0=0.02)[0]
    

    def worker(i):
        time, float = HW.create_path(dt, 15, seed=i)
        swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1]) for x in np.array([time,float]).T])
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        if swap[np.where(time==5)] < 0:
            swap[np.where(time>=5)]=0
        return np.maximum(swap,0), np.minimum(swap,0)

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=cpu_count())(delayed(worker)(i) for i in range(sims))
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
    dump(PE/sims, f'./SimulationData/PE_5Y10YPayer_Swaption_N={sims}_dt={round(dt,2)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_5Y10YPayer_Swaption_N={sims}_dt={round(dt,2)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_5Y10YPayer_Swaption_N={sims}_dt={round(dt,2)}.png', bbox_inches='tight')
    print( 'Finished creating graph')

# Swap with Variation Margin
if False:
    start = timer.time()
    time, float = HW.create_path(dt,10, seed=0)
    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]
    KVM=0 # Threshold for VM
    MTA=0 # Minimum Transfer Amount
    lag=2 # Lookback lag

    def worker(i):
        time, float = HW.create_path(dt,10, seed=i)
        swap = np.array([HW.swap(x[0], S, T, K=K, initRate=x[1]) for x in np.array([time,float]).T])
        
        VM = []

        for t in range(len(time)):
            ttilde = t-lag
            if (t == 0) or (ttilde <= 0):
                VM.append(0)
                continue
            if t == len(time)-1:
                VM.append(0)
                break

            VMhat = VM[ttilde-1]/HW.ZCB(time[ttilde-1], time[ttilde], initRate=float[ttilde-1]) # VM hat in eq 3.2 everything...
            VMta  = VMhat #(ta: to append) 
            
            VMta += (abs(pos(swap[ttilde]-KVM)-pos(VMhat))>MTA)*(pos(swap[ttilde]-KVM)-pos(VMhat))
            VMta += (abs(neg(swap[ttilde]+KVM)-neg(VMhat))>MTA)*(neg(swap[ttilde]+KVM)-neg(VMhat))

            VM.append(VMta)
            print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return np.maximum(swap-VM,0), np.minimum(swap-VM,0)
        
        

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=cpu_count())(delayed(worker)(i) for i in range(sims))
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
    dump(PE/sims, f'./SimulationData/PE_10Y_Swap_With_VM_N={sims}_dt={round(dt,2)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_10Y_Swap_With_VM_N={sims}_dt={round(dt,2)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_10YPayer_Swap_With_VM_N={sims}_dt={round(dt,2)}.png', bbox_inches='tight')
    print( 'Finished creating graph')


    # Swaption with Variation Margin
if False:
    start = timer.time()
    #5Y10Y Payer Swaption Exposure
    time, float = HW.create_path(dt,15, seed=0)
    from joblib import Parallel, delayed, cpu_count
    K=fsolve(lambda x: HW.swap(0, S+5, T+5, x), x0=0.02)[0]
    KVM=0 # Threshold for VM
    MTA=0 # Minimum Transfer Amount
    lag=2 # Lookback lag

    def worker(i):
        time, float = HW.create_path(dt,15, seed=i)
        swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1]) for x in np.array([time,float]).T])
        if swap[np.where(time==5)] < 0:
            swap[np.where(time>=5)]=0
        
        VM = []

        for t in range(len(time)):
            ttilde = t-lag
            if (t == 0) or (ttilde <= 0):
                VM.append(0)
                continue
            if t == len(time)-1:
                VM.append(0)
                break

            VMhat = VM[ttilde-1]/HW.ZCB(time[ttilde-1], time[ttilde], initRate=float[ttilde-1]) # VM hat in eq 3.2 everything...
            VMta  = VMhat #(ta: to append) 
            
            VMta += (abs(pos(swap[ttilde]-KVM)-pos(VMhat))>MTA)*(pos(swap[ttilde]-KVM)-pos(VMhat))
            VMta += (abs(neg(swap[ttilde]+KVM)-neg(VMhat))>MTA)*(neg(swap[ttilde]+KVM)-neg(VMhat))

            VM.append(VMta)
            print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return np.maximum(swap-VM,0), np.minimum(swap-VM,0)

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=cpu_count())(delayed(worker)(i) for i in range(sims))
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

    plt.rc('font',family='Times New Roman')
    float_x = np.arange(5.5,15.5,0.5)
    float_y = np.full((len(float_x),), -2.6)
    fix_x = np.arange(6,16)
    fix_y = np.full(len(fix_x), -2)

    fig, ax = plt.subplots()
    sns.lineplot(x=time, y=PE/sims*100, label = 'EPE')
    sns.lineplot(x=time, y=NE/sims*100, label = 'ENE')
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
    ax.yaxis.set_label_coords(-0.03, 0.5)
    ax.axhline(y=0, color='k', alpha = 0.25)
    plt.grid(alpha = 0.25)
    plt.xticks(np.arange(0, 16))
    plt.legend(frameon = False, fontsize = 18, loc='upper right')
    dump(PE/sims, f'./SimulationData/PE_5Y10Y_Swaption_With_VM_N={sims}_dt={round(dt,2)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_5Y10Y_Swaption_With_VM_N={sims}_dt={round(dt,2)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_5Y10YPayer_Swaption_With_VM_N={sims}_dt={round(dt,2)}.png', bbox_inches='tight')
    print( 'Finished creating graph')