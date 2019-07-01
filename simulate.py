import numpy as np
from numba import njit,prange

# consav package
from consav import linear_interp # for linear interpolation

@njit
def simulate(simT,par,sol,m,delta,c,m0s,tau0,delta0,psi,xi):

    # parallel over initial m
    for j in prange(m0s.size):

        # time loop
        for t in range(simT):
            
            tau = np.fmax(tau0-t,0)
            
            # i. initialize
            if t == 0:
                delta[j,t,:] = delta0
                m[j,t,:] = m0s[j]
                
            # ii. choice
            linear_interp.interp_2d_vec(
                par.grid_delta,par.grid_m,sol.c[tau,:,:],
                delta[j,t,:],m[j,t,:],c[j,t,:])

            # iii. next-period
            if t < simT-1:

                delta[j,t+1,:] = delta[j,t,:]/(par.G*psi[t+1,:])
                a = m[j,t,:] - c[j,t,:]
                m[j,t+1,:] = par.R*a/(par.G*psi[t+1,:]) + xi[t+1,:]