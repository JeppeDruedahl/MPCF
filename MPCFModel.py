# -*- coding: utf-8 -*-
"""MPCF

Solves the Deaton-Carroll buffer-stock consumption model with an anticipated income shock.
"""

##############
# 1. imports #
##############


import time
import numpy as np

# consav package
from consav import ModelClass,jit # baseline model class and jit
from consav.grids import nonlinspace # grids
from consav.quadrature import create_PT_shocks # income shocks

# local modules
import utility
import post_decision
import egm
import simulate

############
# 2. model #
############

class MPCFClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def settings(self):

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. list not-floats for safe type inference
        self.not_floats = ['Npsi','Nxi','Ndelta','Nm','Na','Ntau','Nshocks','max_iter','do_print']

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. preferences
        par.beta = 0.96**(1/12)
        par.rho = 2.0

        # b. returns and income
        par.R = 1.04**(1/12)
        par.G = 1.03**(1/12)
        par.sigma_psi = 0.02122033
        par.Npsi = 6
        par.sigma_xi = 0.30467480
        par.Nxi = 6
        par.pi = 0.0
        par.mu = np.nan

        # c. extra income        
        par.zeta = 1.0
        par.Ntau = 12
        
        # d. grids (number of points)
        par.Ndelta = 50
        par.Nm = 20_000
        par.m_mid = 120.0
        par.m_max = 100_000.0
        par.Na = 10_000
        par.a_mid = par.m_mid
        par.a_max = par.m_max

        # e. misc
        par.max_iter = 10_000        
        par.tol = 1e-6
        par.do_print = True

    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        self.setup_grids()
        self._solve_prep()
        
    def setup_grids(self):
        """ construct grids for states and shocks """

        par = self.par

        # a. states (unequally spaced vectors of length Na)
        par.grid_delta = np.zeros(par.Ndelta)
        if par.Ndelta > 1:
            par.grid_delta[1] = 1e-4
            par.grid_delta[2:] = nonlinspace(2*1e-4,0.1,par.Ndelta-2,1.3)
            
        m_min = 0
        if np.isnan(par.m_mid):
            par.grid_m = nonlinspace(m_min,par.m_max,par.Nm,1.2)
        else:
            Nm_base = par.Nm//2
            par.grid_m = np.zeros(par.Nm)
            par.grid_m[:Nm_base] = nonlinspace(m_min,par.m_mid,Nm_base,1.1)
            par.grid_m[Nm_base-1:] = nonlinspace(par.m_mid,par.m_max,par.Nm-Nm_base+1,1.1)

        a_min = m_min+1e-6
        if np.isnan(par.a_mid):
            par.grid_a = nonlinspace(a_min,par.a_max,par.Na,1.2)
        else:
            Na_base = par.Na//2
            par.grid_a = np.zeros(par.Na)
            par.grid_a[:Na_base] = nonlinspace(a_min,par.a_mid,Na_base,1.1)
            par.grid_a[Na_base-1:] = nonlinspace(par.a_mid,par.a_max,par.Na-Na_base+1,1.1)

        # b. shocks (qudrature nodes and weights using GaussHermite)
        par.Nxi = 1 if par.sigma_xi == 0 else par.Nxi
        par.Npsi = 1 if par.sigma_psi == 0 else par.Npsi

        shocks = create_PT_shocks(par.sigma_psi,par.Npsi,par.sigma_xi,par.Nxi,par.pi,par.mu)
        par.psi,par.psi_w,par.xi,par.xi_w,par.Nshocks = shocks

        # c. perfect foresight
        par.MPC_PF = 1-(par.R*par.beta)**(1/par.rho)/par.R
        par.MPCP_PF = par.MPC_PF/(1-1/par.R)
        par.MPCF_PF = np.zeros(par.Ntau+1)
        for tau in range(par.Ntau+1):
            par.MPCF_PF[tau] = par.R**(-tau)*par.MPC_PF/(1-par.zeta/par.R)
        
    #########
    # solve #
    #########

    def _solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol

        sol.c = np.zeros((par.Ntau+1,par.Ndelta,par.Nm))
        sol.q = np.zeros((par.Ntau+1,par.Ndelta,par.Na))

    def solve(self):
        
        par = self.par

        # a. solve directly
        if par.Ndelta == 1:
            
            if par.do_print: print('solving full model\n')
            self.solve_with_c0()
        
        # b. solve in two steps
        else:

            # i. without delta dimension
            if par.do_print: print('solving model with Ndelta=1 and Ntau=1\n')

            Ntau = par.Ntau
            Ndelta = par.Ndelta
            tol = par.tol
            Npsi = par.Npsi
            Nxi = par.Nxi
            par.Ntau = 1
            par.Ndelta = 1
            par.tol = 1e-8
            par.Npsi = 3
            par.Nxi = 3
            self.setup_grids()

            self.solve_with_c0()
            c0 = self.sol.c[0,:,:].copy()
            for idelta in range(par.Ndelta):
                c0[idelta,:] += par.MPC_PF*1/(1-par.zeta*1/par.R)*par.grid_delta[idelta]

            par.Ntau = Ntau
            par.Ndelta = Ndelta
            par.tol = tol
            par.Npsi = Npsi
            par.Nxi = Nxi
            self.setup_grids()

            if par.do_print: print('')

            # ii. full
            if par.do_print: print('solving full model\n')

            self.solve_with_c0(c0=c0)

    def solve_with_c0(self,c0=np.array([])):
        """ solve the model using egm and the initial guess in c0 """

        # a. allocate solution
        self._solve_prep()
        
        # b. backwards induction until convergence
        with jit(self) as model:

            par = model.par
            sol = model.sol

            it = 0
            max_abs_diff = np.inf
            while it < par.max_iter:

                # i. first iteration
                if it == 0:
                    
                    tic = time.time()

                    if c0.size == 0:
                        for idelta in range(par.Ndelta):
                            b = par.grid_m-1
                            h = 1/(1-par.G/par.R) + 1/(1-par.zeta*1/par.R)*par.grid_delta[idelta]
                            sol.c[0,idelta,:] = np.fmin(par.MPC_PF*(b+h),par.grid_m)
                    elif c0.size:
                        sol.c[0,:,:] = c0

                # ii. all other iteration
                else:
                    
                    # o. compute post-decision functions
                    post_decision.compute_q(sol,par,0,0)

                    # oo. solve bellman equation
                    egm.solve_bellman(sol,par,0)                    
                
                # iii. check convergence
                if it > 0:
                    max_abs_diff = np.max(np.abs(sol.c[0]-c_old))

                # iv. save old consumption function
                c_old = sol.c[0].copy()

                # v. print
                toc = time.time()

                if it > 0 and it%50 == 0 and par.do_print: 
                    print(f'{it:4d}: max_abs_diff = {max_abs_diff:12.8f} (elapsed: {toc-tic:5.1f} secs)')   
                
                if max_abs_diff < par.tol:
                    print('-> convergence achieved')
                    break
                
                it += 1

            # c. bakcwards induction through the anticipation horizon
            for itau in range(1,par.Ntau+1):

                # a. compute post-decision functions
                post_decision.compute_q(sol,par,itau,itau-1)

                # b. solve bellman equation
                egm.solve_bellman(sol,par,itau)       

    ############
    # simulate #
    ############
    
    def simulate(self,simN,simT,m0s,tau0,delta0,seed=1917,postfix=''):

        with jit(self) as model:

            par = model.par
            sol = model.sol

            # a. allocate
            shape =(m0s.size,simT,simN)
            m = np.zeros(shape)
            delta = np.zeros(shape)
            c = np.zeros(shape)
            P = np.zeros(shape)
            C = np.zeros(shape)

            # b. draw random
            np.random.seed(1917)
            psi = np.exp(np.random.normal(loc=-0.5*par.sigma_psi**2,scale=par.sigma_psi,size=(simT,simN)))
            xi = np.exp(np.random.normal(loc=-0.5*par.sigma_xi**2,scale=par.sigma_xi,size=(simT,simN)))
            
            # c. simulate
            simulate.simulate(simT,par,sol,m,delta,c,P,C,m0s,tau0,delta0,psi,xi)

        return m,c,C
     