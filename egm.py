import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import utility

@njit(parallel=True)
def solve_bellman(sol,par,t):
    """solve the bellman equation using the endogenous grid method"""
    
    # unpack (helps numba optimize)
    c = sol.c

    # loop over delta state
    for idelta in prange(par.Ndelta):
        
        # temp
        m_temp = np.zeros(par.Na+1)
        c_temp = np.zeros(par.Na+1)

        # a. find consumption
        for ia in range(par.Na):
            c_temp[ia+1] = utility.inv_marg_func(sol.q[t,idelta,ia],par)
            m_temp[ia+1] = par.grid_a[ia]+c_temp[ia+1]
            if t == 0:
                m_temp[ia+1] -= par.grid_delta[idelta]
        
        if t == 0: # assuming the borrowing constraint is binding for low enough m
           c_temp[0] = par.grid_delta[idelta]

        # b. interpolate to common grid
        linear_interp.interp_1d_vec_mon_noprep(m_temp,c_temp,par.grid_m,c[t,idelta,:])