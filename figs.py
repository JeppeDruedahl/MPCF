import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("seaborn-whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

def _cfunc(model):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # c. plot consumption
    for idelta in range(0,par.Ndelta,10):    
        ax.plot(par.grid_m,sol.c[0,idelta,:],ls='-',label=f'$\delta_t = {par.grid_delta[idelta]:.2f}$')

    # d. details
    ax.set_xlabel('cash-on-hand, $m_t$')
    ax.set_ylabel('consumption, $c_t$')
    
    if par.Ndelta > 1:
        ax.legend(frameon=True)
        
    return fig,ax

def cfunc(model,m_max=12,c_max=1.5,postfix='',savefig=False):
    
    # a. zoom
    fig,ax = _cfunc(model)
    ax.set_xlim([0,m_max])
    ax.set_ylim([0,c_max])

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/cfunc{postfix}.pdf')

    # a. zoom
    fig,ax = _cfunc(model)
    
    fig.tight_layout()
    if savefig: fig.savefig(f'figs/cfunc_convergence{postfix}.pdf')

def _MPC(model):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # c. plot consumption
    MPC = np.diff(sol.c[0,0,:])/np.diff(par.grid_m)
    ax.plot(par.grid_m[:-1],MPC,ls='-')

    # d. details
    ax.set_xlabel('cash-on-hand, $m_t$')
    ax.set_ylabel('MPC')
    
    ax.axhline(par.MPC_PF,ls='--',lw=1,color=colors[0])
    
    return fig,ax

def MPC(model,m_max=12,postfix='',savefig=False):

    # a. zoom
    fig,ax = _MPC(model)
    ax.set_yscale('log')
    ax.set_xlim([model.par.grid_m[0],m_max])
    ax.set_ylim([0.001,1.1])

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/MPC{postfix}.pdf')

    # b. convergence
    fig,ax = _MPC(model)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.1,model.par.grid_m[-1]])
    ax.set_ylim([0.001,1.1])

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/MPC_convergence{postfix}.pdf')

def _MPCF(model,taus=[0,1,4,12],show_theoretical=False):
    
    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # c. plot consumption
    for tau in taus:
        MPCF = (sol.c[tau,5,:]-sol.c[tau,0,:])/(par.grid_delta[5]-par.grid_delta[0])    
        ax.plot(par.grid_m,MPCF,label=f'$\\tau_t = {tau}$')

    # d. details
    ax.set_xlabel('cash-on-hand, $m_t$')
    ax.set_ylabel('MPCF')
    
    if show_theoretical:
        for i,tau in enumerate(taus):
            ax.axhline(par.MPCF_PF[tau],ls='--',lw=1,color=colors[i],label='')
    
    ax.legend(frameon=True)
    
    return fig,ax    

def MPCF(model,m_max=12,taus=[0,1,4,12],show_theoretical=False,postfix='',savefig=False):

    # a. zoom
    fig,ax = _MPCF(model,taus=taus,show_theoretical=show_theoretical)
    ax.set_xlim([0,m_max])

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/MPCF{postfix}.pdf')

    # b. convergence
    fig,ax = _MPCF(model,taus=taus,show_theoretical=True)
    ax.set_xscale('log')
    ax.set_xlim([0.1,model.par.grid_m[-1]])

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/MPCF_convergence{postfix}.pdf')

def _buffer(model,freq=1,do_print=False):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # c. plot
    m = par.grid_m
    c = sol.c[0,0,:]
    
    ax.plot(m/freq,c,ls='-',label=f'$c_t$')
    
    a = m-c
    ax.plot(m/freq,a/freq,ls='-',label=f'$a_t$')
    
    exp_m_plus = par.R*a + 1
    
    ax.plot(m/freq,exp_m_plus/freq,ls='-',label=f'$\mathbb{{E}}_t[Ra_t + \\xi_{{t+1}}]$')
    ax.plot(m/freq,m/freq,label='')
    
    # d. target
    i_target = (np.abs(exp_m_plus - m)).argmin()
    target = par.grid_m[i_target]
    ax.axvline(target,ls='--',lw=1,color='black')
    
    if do_print: print(f'buffer-stock target, a: {a[i_target]:.2f}')
    
    # e. details
    ax.set_xlabel('cash-on-hand, $m_t$')
    ax.legend(frameon=True)
        
    return fig,ax

def buffer(model,freq=1,postfix='',savefig=False,do_print=False):

    fig,ax = _buffer(model,freq=freq,do_print=do_print)
    ax.set_xlim([0,5])
    ax.set_ylim([0,3])

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/buffer{postfix}.pdf')    

def simulate(model,savefig=False,postfix=''):

    # a. settings
    m0s = np.array([0.75,1.5,2,4,12])
    tau0 = 4
    delta0 = 0.01
    simN = 100_000
    simT = 16

    # b. simulate
    _delta,m_before,c_before = model.simulate(simN,simT,m0s,tau0,0)
    _delta,m_after,c_after = model.simulate(simN,simT,m0s,tau0,delta0)    
    
    # c. figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for j,m0 in enumerate(m0s):
        ax.plot(np.mean(c_after[j,:,:]-c_before[j,:,:],axis=1)/delta0,
                ls='-',marker='o',markersize=4,
                label=f'$m_0 = {m0:.2f}$')

    ax.axvline(tau0,ls='--',lw=1,color='black')
    
    ax.set_xlabel('time, $t$')
    ax.set_ylabel('dynamic MPCF')
    ax.legend(frameon=True)

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/simulation{postfix}.pdf')

    # d. figure - still constrained
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # full
    j = 0
    ax.plot(np.mean(c_after[j,:,:]-c_before[j,:,:],axis=1)/delta0,
            ls='-',marker='o',markersize=4,color=colors[0],
            label=f'$m_0 < {m0s[0]:.2f}$')

    # conditional

    cutoff = 1
    I = (m_before[j,tau0-1,:] < cutoff)
    c_diff = np.zeros(simT)
    for t in range(simT):
        c_diff[t] = np.mean(c_after[j,t,I])-np.mean(c_before[j,t,I])
        
    ax.plot(c_diff/delta0,
            ls='--',marker='o',markersize=4,color=colors[0],
            label=f'$m_0 < {m0s[0]:.2f}, m_{{\\tau_0-1}} < {cutoff:.2f}$')

    ax.axvline(tau0,ls='--',lw=1,color='black')
    
    ax.set_xlabel('time, $t$')
    ax.set_ylabel('dynamic MPCF')
    ax.legend(frameon=True)

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/simulation{postfix}_below_cutoff.pdf')        
