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
    
    fig,ax = _cfunc(model)
    ax.set_xlim([0,m_max])
    ax.set_ylim([0,c_max])
    ax.set_xticks(np.arange(0,m_max+1,1))

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/cfunc{postfix}.pdf')

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
    ax.text(0.25,1.1*par.MPC_PF,'PIH') 

    return fig,ax

def MPC(model,m_max=12,postfix='',savefig=False):

    # a. zoom
    fig,ax = _MPC(model)
    ax.set_yscale('log')
    ax.set_xlim([model.par.grid_m[0],m_max])
    ax.set_ylim([0.001,1.1])
    ax.set_xticks(np.arange(0,m_max+1,1))

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

def _MPCF(model,taus=[0,1,4,12],show_analytical=False):
    
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
    
    if show_analytical:
        for i,tau in enumerate(taus):
            ax.axhline(par.MPCF_PF[tau],ls='--',lw=1,color=colors[i],label='')
        
        ax.text(0.25,par.MPCF_PF[taus[-1]]-0.15,'PIH$\Rightarrow$',rotation=90) 

    ax.legend(frameon=True)

    return fig,ax    

def MPCF(model,m_max=12,taus=[1,4,6,8],show_analytical=True,postfix='',savefig=False):

    # a. zoom
    fig,ax = _MPCF(model,taus=taus,show_analytical=show_analytical)
    ax.set_xlim([0,m_max])
    ax.set_xticks(np.arange(0,m_max+1,1))
    ax.set_ylim([0,1.3])
    
    fig.tight_layout()
    if savefig: fig.savefig(f'figs/MPCF{postfix}.pdf')

    # b. convergence
    fig,ax = _MPCF(model,taus=taus,show_analytical=True)
    ax.set_xscale('log')
    ax.set_xlim([0.1,model.par.grid_m[-1]])
    ax.set_ylim([0,1.3])

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/MPCF_convergence{postfix}.pdf')

def simulate(model,savefig=False,postfix=''):

    # a. settings
    m0s = np.array([0.75,1.5,2,4,12])
    tau0 = 6
    delta0 = 0.05
    simN = 100_000
    simT = 31

    # b. simulate
    m_before,c_before,C_before = model.simulate(simN,simT,m0s,tau0,0)
    m_after,c_after,C_after = model.simulate(simN,simT,m0s,tau0,delta0)    
    
    # c. figure - response
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    time = np.arange(-tau0,-tau0+simT,1)
    xticks = np.arange(-tau0,-tau0+simT,2)

    for j,m0 in enumerate(m0s):
        ax.plot(time,np.mean(C_after[j,:,:]-C_before[j,:,:],axis=1)/delta0,
                ls='-',marker='o',markersize=4,
                label=f'$M_0 = {m0:.2f}$')

    ax.axvline(0,ls='--',lw=1,color='black')
    ax.text(0.25,0.1,'$\Leftarrow$cash-flow arrives') 

    ax.set_xlabel('time relative to arrival of cash-flow')
    ax.set_ylabel('dynamic MPCF')
    ax.set_xticks(xticks)
    ax.legend(frameon=True)

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/simulation{postfix}.pdf')
    
    # d. figure - still constrained
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # before
    j = 0
    ax.plot(time,np.mean(c_before[j,:,:] >= 0.99*m_before[j,:,:],axis=1),
            ls='-',marker='o',markersize=4,color=colors[0],
            label=f'$\Delta = 0.00$')

    # after
    ax.plot(time,np.mean(c_after[j,:,:] >= 0.99*m_after[j,:,:],axis=1),
            ls='-',marker='o',markersize=4,color=colors[1],
            label=f'$\Delta = {delta0:.2f}$')

    ax.axvline(0,ls='--',lw=1,color='black')
    
    ax.set_xlabel('time relative to arrival of cash-flow')
    ax.set_ylabel('constrained, share')
    ax.set_xticks(xticks)
    ax.legend(frameon=True)

    fig.tight_layout()
    if savefig: fig.savefig(f'figs/simulation{postfix}_constrained.pdf')        