# -*- coding: utf-8 -*-
"""MPCF

Solves the Deaton-Carroll buffer-stock consumption model with an anticipated income shock.
"""

##############
# 1. imports #
##############

import yaml
yaml.warnings({'YAMLLoadWarning': False})

import time
import numpy as np
from numba import boolean, int64, double

# consav package
from consav import misc # various tools
from consav import ModelClass # baseline model class

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
    
    def __init__(self,name='baseline',load=False,**kwargs):
        """ basic setup

        Args:

            name (str,optional): name, used when saving/loading
            load (bool,optinal): load from disc
             **kwargs: change to baseline parameter in .par
            
        Define parlist, sollist and simlist contain information on the
        model parameters and the variables when solving and simulating.

        Call .setup(**kwargs).

        """        

        self.name = name 
        self.solmethod = 'egm'
        
        # a. define subclasses
        parlist = [ # (name,numba type), parameters, grids etc.
            ('beta',double), 
            ('rho',double),
            ('R',double),
            ('G',double),
            ('sigma_psi',double),
            ('Npsi',int64),
            ('sigma_xi',double),
            ('Nxi',int64),
            ('pi',double),
            ('mu',double),
            ('Ndelta',int64),
            ('grid_delta',double[:]),             
            ('Nm',int64),
            ('m_mid',double),
            ('m_max',double),
            ('grid_m',double[:]),        
            ('Na',int64),
            ('a_mid',double),
            ('a_max',double),            
            ('grid_a',double[:]),        
            ('Ntau',int64),        
            ('zeta',double),        
            ('Nshocks',int64),        
            ('psi',double[:]),        
            ('psi_w',double[:]),        
            ('xi',double[:]),        
            ('xi_w',double[:]),        
            ('max_iter',int64),
            ('tol',double),
            ('MPC_PF',double),
            ('MPCP_PF',double),
            ('MPCF_PF',double[:]),
            ('do_print',boolean), # boolean
        ]
        
        sollist = [ # (name, numba type), solution data
            ('c',double[:,:,:]),
            ('q',double[:,:,:])
        ]        

        simlist = [ # (name, numba type), simulation data
            ('m',double[:,:,]),
            ('c',double[:,:,]),
            ('delta',double[:,:,]),
            ('psi',double[:,:,]),
            ('xi',double[:,:,])
        ]      

        # b. create subclasses
        self.par,self.sol,self.sim = self.create_subclasses(parlist,sollist,simlist)

        # note: the above returned classes are in a format where they can be used in numba functions

        # c. load
        if load:
            self.load()
        else:
            self.setup(**kwargs)

    def setup(self,**kwargs):
        """ define baseline values and update with user choices

        Args:

             **kwargs: change to baseline parameters in .par

        """   

        # a. baseline parameters
        
        # preferences
        self.par.beta = 0.96**(1/12)
        self.par.rho = 2

        # returns and income
        self.par.R = 1.04**(1/12)
        self.par.G = 1.03**(1/12)
        self.par.sigma_psi = 0.02122033
        self.par.Npsi = 6
        self.par.sigma_xi = 0.30467480
        self.par.Nxi = 6
        self.par.pi = 0.0
        self.par.mu = np.nan

        # extra income        
        self.par.zeta = 1
        self.par.Ntau = 12
        
        # grids (number of points)
        self.par.Ndelta = 50
        self.par.Nm = 20_000
        self.par.m_mid = 120
        self.par.m_max = 100_000
        self.par.Na = 10_000
        self.par.a_mid = self.par.m_mid
        self.par.a_max = self.par.m_max

        # misc
        self.par.max_iter = 10_000        
        self.par.tol = 1e-6
        self.par.do_print = True

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val
        
        # c. setup_grids
        self.setup_grids()
        
    def setup_grids(self):
        """ construct grids for states and shocks """

        # a. states (unequally spaced vectors of length Na)
        self.par.grid_delta = np.zeros(self.par.Ndelta)
        if self.par.Ndelta > 1:
            self.par.grid_delta[1] = 1e-4
            self.par.grid_delta[2:] = misc.nonlinspace(2*1e-4,0.1,self.par.Ndelta-2,1.3)
            
        m_min = 0
        if np.isnan(self.par.m_mid):
            self.par.grid_m = misc.nonlinspace(m_min,self.par.m_max,self.par.Nm,1.2)
        else:
            Nm_base = self.par.Nm//2
            self.par.grid_m = np.zeros(self.par.Nm)
            self.par.grid_m[:Nm_base] = misc.nonlinspace(m_min,self.par.m_mid,Nm_base,1.1)
            self.par.grid_m[Nm_base-1:] = misc.nonlinspace(self.par.m_mid,self.par.m_max,self.par.Nm-Nm_base+1,1.1)

        a_min = m_min+1e-6
        if np.isnan(self.par.a_mid):
            self.par.grid_a = misc.nonlinspace(a_min,self.par.a_max,self.par.Na,1.2)
        else:
            Na_base = self.par.Na//2
            self.par.grid_a = np.zeros(self.par.Na)
            self.par.grid_a[:Na_base] = misc.nonlinspace(a_min,self.par.a_mid,Na_base,1.1)
            self.par.grid_a[Na_base-1:] = misc.nonlinspace(self.par.a_mid,self.par.a_max,self.par.Na-Na_base+1,1.1)

        # b. shocks (qudrature nodes and weights using GaussHermite)
        self.par.Nxi = 1 if self.par.sigma_xi == 0 else self.par.Nxi
        self.par.Npsi = 1 if self.par.sigma_psi == 0 else self.par.Npsi

        shocks = misc.create_shocks(self.par.sigma_psi,self.par.Npsi,self.par.sigma_xi,self.par.Nxi,self.par.pi,self.par.mu)
        self.par.psi,self.par.psi_w,self.par.xi,self.par.xi_w,self.par.Nshocks = shocks

        # c. perfect foresight
        self.par.MPC_PF = 1-(self.par.R*self.par.beta)**(1/self.par.rho)/self.par.R
        self.par.MPCP_PF = self.par.MPC_PF/(1-1/self.par.R)
        self.par.MPCF_PF = np.zeros(self.par.Ntau+1)
        for tau in range(self.par.Ntau+1):
            self.par.MPCF_PF[tau] = self.par.R**(-tau)*self.par.MPC_PF/(1-self.par.zeta/self.par.R)
        
    #########
    # solve #
    #########

    def _solve_prep(self):
        """ allocate memory for solution """

        self.sol.c = np.zeros((self.par.Ntau+1,self.par.Ndelta,self.par.Nm))
        self.sol.q = np.zeros((self.par.Ntau+1,self.par.Ndelta,self.par.Na))

    def solve(self):
        
        # a. solve directly
        if self.par.Ndelta == 1:
            
            if self.par.do_print: print('solving full model\n')
            self.solve_with_c0()
        
        # b. solve in two steps
        else:

            # i. without delta dimension
            if self.par.do_print: print('solving model with Ndelta=1 and Ntau=1\n')

            Ntau = self.par.Ntau
            Ndelta = self.par.Ndelta
            tol = self.par.tol
            Npsi = self.par.Npsi
            Nxi = self.par.Nxi
            self.par.Ntau = 1
            self.par.Ndelta = 1
            self.par.tol = 1e-8
            self.par.Npsi = 3
            self.par.Nxi = 3
            self.setup_grids()

            self.solve_with_c0()
            c0 = self.sol.c[0,:,:].copy()
            for idelta in range(self.par.Ndelta):
                c0[idelta,:] += self.par.MPC_PF*1/(1-self.par.zeta*1/self.par.R)*self.par.grid_delta[idelta]

            self.par.Ntau = Ntau
            self.par.Ndelta = Ndelta
            self.par.tol = tol
            self.par.Npsi = Npsi
            self.par.Nxi = Nxi
            self.setup_grids()

            if self.par.do_print: print('')

            # ii. full
            if self.par.do_print: print('solving full model\n')

            self.solve_with_c0(c0=c0)

    def solve_with_c0(self,c0=np.array([])):
        """ solve the model using egm and the initial guess in c0 """

        # a. allocate solution
        self._solve_prep()
        
        # b. backwards induction until convergence
        it = 0
        max_abs_diff = np.inf
        while it < self.par.max_iter:

            # i. first iteration
            if it == 0:
                
                tic = time.time()

                if c0.size == 0:
                    for idelta in range(self.par.Ndelta):
                        b = self.par.grid_m-1
                        h = 1/(1-self.par.G/self.par.R) + 1/(1-self.par.zeta*1/self.par.R)*self.par.grid_delta[idelta]
                        self.sol.c[0,idelta,:] = np.fmin(self.par.MPC_PF*(b+h),self.par.grid_m)
                elif c0.size:
                    self.sol.c[0,:,:] = c0

            # ii. all other iteration
            else:
                
                # o. compute post-decision functions
                post_decision.compute_q(self.sol,self.par,0,0)

                # oo. solve bellman equation
                egm.solve_bellman(self.sol,self.par,0)                    
            
            # iii. check convergence
            if it > 0:
                max_abs_diff = np.max(np.abs(self.sol.c[0]-c_old))

            # iv. save old consumption function
            c_old = self.sol.c[0].copy()

            # v. print
            toc = time.time()

            if it > 0 and it%50 == 0 and self.par.do_print: 
                print(f'{it:4d}: max_abs_diff = {max_abs_diff:12.8f} (elapsed: {toc-tic:5.1f} secs)')   
            
            if max_abs_diff < self.par.tol:
                print('-> convergence achieved')
                break
            
            it += 1

        # c. bakcwards induction through the anticipation horizon
        for itau in range(1,self.par.Ntau+1):

            # a. compute post-decision functions
            post_decision.compute_q(self.sol,self.par,itau,itau-1)

            # b. solve bellman equation
            egm.solve_bellman(self.sol,self.par,itau)       

    ############
    # simulate #
    ############
    
    def simulate(self,simN,simT,m0s,tau0,delta0,seed=1917,postfix=''):

        # a. allocate
        shape =(m0s.size,simT,simN)
        m = np.zeros(shape)
        delta = np.zeros(shape)
        c = np.zeros(shape)
        P = np.zeros(shape)
        C = np.zeros(shape)

        # b. draw random
        np.random.seed(1917)
        psi = np.exp(np.random.normal(loc=-0.5*self.par.sigma_psi**2,scale=self.par.sigma_psi,size=(simT,simN)))
        xi = np.exp(np.random.normal(loc=-0.5*self.par.sigma_xi**2,scale=self.par.sigma_xi,size=(simT,simN)))
        
        # c. simulate
        simulate.simulate(simT,self.par,self.sol,m,delta,c,P,C,m0s,tau0,delta0,psi,xi)

        return m,c,C
     