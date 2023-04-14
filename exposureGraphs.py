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

beta = np.array([1.879892, 0.579211, 	3.993992, 1.445091])
tau  = np.array([ 16.633491, 	0.319680])
reversion=0.13949636660880768 
volatility=0.017793899652989272
# HW = HullWhite(initial=0.02459103, reversion=0.03, volatility=0.00200, Gamma=1000, b=beta, tau=tau)
HW = HullWhite(initial=0.02459103, reversion=reversion, volatility=volatility, Gamma=1000, b=beta, tau=tau)




#Setting up tenor
T=np.arange(0,10+0.5,0.5)
S=np.arange(0,11,1)

# Other settings
dt = 1/365
sims = 1000

#10Y Payer Swap Exposure
if False:
    print('10Y Payer Swap Exposure')
    start = timer.time()
    time, float = HW.create_path(dt,10, seed=0)
    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]

    def worker(i):
        time, float = HW.create_path(dt,10, seed=i)
        swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return np.maximum(swap,0), np.minimum(swap,0)

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=np.minimum(cpu_count(),sims))(delayed(worker)(i) for i in range(sims))
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
if False:
    print('5Y10YForward Swap Exposure')
    start = timer.time()
    time, float = HW.create_path(dt,15, seed=0)
    K=fsolve(lambda x: HW.swap(0, S+5, T+5, x), x0=0.02)[0]


    def worker(i):
        time, float = HW.create_path(dt,15, seed=i)
        swap = np.array([HW.swapextended(x[0], S+5, T+5, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        return np.maximum(swap,0), np.minimum(swap,0)

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=np.minimum(cpu_count(),sims))(delayed(worker)(i) for i in range(sims))
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

#5Y10Y Payer Swaption Exposure
if False:
    print('5Y10Y Payer Swaption Exposure')
    start = timer.time()
    time, float = HW.create_path(dt,15, seed=0)
    from joblib import Parallel, delayed, cpu_count
    K=fsolve(lambda x: HW.swap(0, S+5, T+5, x), x0=0.02)[0]
    

    def worker(i):
        time, float = HW.create_path(dt, 15, seed=i)
        swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
        print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
        if swap[np.where(time==5)] < 0:
            swap[np.where(time>=5)]=0
        return np.maximum(swap,0), np.minimum(swap,0)

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=np.minimum(cpu_count(),sims))(delayed(worker)(i) for i in range(sims))
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
    dump(PE/sims, f'./SimulationData/PE_5Y10YPayer_Swaption_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_5Y10YPayer_Swaption_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_5Y10YPayer_Swaption_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')

# Swap with Variation Margin
if True:
    print('10Y Payer Swap with Variation Margin')
    start = timer.time()
    time, float = HW.create_path(dt,10, seed=0)
    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]
    KVM=0 # Threshold for VM
    MTA=0 # Minimum Transfer Amount
    lag=2 # Lookback lag

    def worker(i):
        time, float = HW.create_path(dt,10, seed=i)
        swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
        
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

        exposure = np.where(swap > 0, np.maximum(swap-VM,0), np.minimum(swap-VM,0))
        return np.maximum(exposure,0), np.minimum(exposure,0)
        
        

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=np.minimum(cpu_count(),sims))(delayed(worker)(i) for i in range(sims))
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
    dump(PE/sims, f'./SimulationData/PE_10Y_Swap_With_VM_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_10Y_Swap_With_VM_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_10YPayer_Swap_With_VM_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')


    # Swaption with Variation Margin
if False:
    print('5Y10Y Payer Swaption Exposure with Variation Margin')
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
        swap = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
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
    results = Parallel(n_jobs=np.minimum(cpu_count(),sims))(delayed(worker)(i) for i in range(sims))
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

#Swap with VM and IM
if False:
    print('10Y Payer Swap with Variation and Initial Margin')
    start = timer.time()
    time, float = HW.create_path(dt,10, seed=0)
    K=fsolve(lambda x: HW.swap(0, S+5, T+5, x), x0=0.02)[0]
    KVM=0 # Threshold for VM
    MTA=0 # Minimum Transfer Amount
    lag=2 # Lookback lag
    oneBP=0.0001
    RW_6M=53
    T_b=230
    def worker(i):
        time, float = HW.create_path(dt,10, seed=i)
        swap = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float, schedule=time, initRate=x[1]) for x in np.array([time,float]).T])
        
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
        
        swapBP = np.array([HW.swapextended(x[0], S, T, K=K, floatRate=float+oneBP, schedule=time, initRate=x[1]+oneBP) for x in np.array([time,float]).T])
        IM = []
        for t in range(len(time)):
                ttilde = t-lag
                if (t == 0) or (ttilde <= 0):
                        IM.append(0)
                        continue
                if t == len(time)-1:
                        IM.append(0)
                        break
                s = swapBP[ttilde]-swap[ttilde]
                CR_B = np.maximum(1,np.sqrt(abs(s)/T_b))
                IM.append(CR_B*s*RW_6M)
                
        result = np.where(swap>0, np.maximum(swap-VM-IM,0), np.minimum(swap-VM+IM,0))

        return np.maximum(result, 0), np.minimum(result, 0)
        
        

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=np.minimum(cpu_count(),sims))(delayed(worker)(i) for i in range(sims))
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
    dump(PE/sims, f'./SimulationData/PE_10Y_Swap_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_10Y_Swap_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_10YPayer_Swap_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')


#Swaption with VM and IM
if False:
    print('5Y Payer Swaption with Variation and Initial Margin')
    start = timer.time()
    time, float = HW.create_path(dt,15, seed=0)
    K=fsolve(lambda x: HW.swap(0, S, T, x), x0=0.02)[0]
    KVM=0 # Threshold for VM
    MTA=0 # Minimum Transfer Amount
    lag=2 # Lookback lag
    oneBP=0.0001
    RW_6M=53
    T_b=230
    def worker(i):
        time, float = HW.create_path(dt, 15, i)
        swaption = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1], floatRate=float, schedule=time) for x in np.array([time,float]).T])
        if swaption[np.where(time==5)] < 0:
            swaption[np.where(time>=5)]=0

        swaptionBP = np.array([HW.swaption(x[0], Te=5, S=S+5, T=T+5, K=K, initRate=x[1]+oneBP, floatRate=float+oneBP, schedule=time,) for x in np.array([time,float]).T])
        if swaptionBP[np.where(time==5)] < 0:
            swaptionBP[np.where(time>=5)]=0
            
        IM = []
        VM = []

        for t in range(len(time)):
            ttilde = t-lag
            if (t == 0) or (ttilde <= 0):
                IM.append(0)
                VM.append(0)
                continue
            if t == len(time)-1:
                IM.append(0)
                VM.append(0)
                break

            VMhat = VM[ttilde-1]/HW.ZCB(time[ttilde-1], time[ttilde], initRate=float[ttilde-1]) # VM hat in eq 3.2 everything...
            VMta  = VMhat #(ta: to append) 
            
            VMta += (abs(pos(swaption[ttilde]-KVM)-pos(VMhat))>MTA)*(pos(swaption[ttilde]-KVM)-pos(VMhat))
            VMta += (abs(neg(swaption[ttilde]+KVM)-neg(VMhat))>MTA)*(neg(swaption[ttilde]+KVM)-neg(VMhat))

            VM.append(VMta)
            s = swaptionBP[ttilde]-swaption[ttilde]
            CR_B = np.maximum(1,np.sqrt(abs(s)/T_b))
            IM.append(CR_B*s*RW_6M)
            print('{:.2f}%'.format(round(i/sims*100, 2)), end='\r')
                        
        exposure = np.where(swaption > 0, np.maximum(swaption-VM-IM,0), np.minimum(swaption-VM+IM,0))
        return np.maximum(exposure, 0), np.minimum(exposure, 0)
        
        

    print('Starting parallel processing with {} cores'.format(cpu_count()))
    results = Parallel(n_jobs=np.minimum(cpu_count(),sims))(delayed(worker)(i) for i in range(sims))
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
    dump(PE/sims, f'./SimulationData/PE_5Y_Swaption_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.joblib')
    dump(NE/sims, f'./SimulationData/NE_5Y_Swaption_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.joblib')
    plt.savefig(f'./Graphs/Exposure_Plot_5YPayer_Swaption_With_VM_AND_IM_N={sims}_dt={int(1/dt)}.png', bbox_inches='tight')
    print( 'Finished creating graph')