import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation

# local modules
import utility

@njit(parallel=True)
def compute_q(sol,par,t,t_plus):
    """ compute the post-decision function q """

    # unpack (helps numba optimize)
    q = sol.q

    # loop over delta
    for idelta in prange(par.Ndelta):

        # clean-up
        q[t,idelta,:] = 0

        # temp
        m_plus = np.empty(par.Na)
        c_plus = np.empty(par.Na)

        # loop over shock
        for ishock in range(par.Nshocks):
            
            # i. shocks
            psi = par.psi[ishock]
            psi_w = par.psi_w[ishock]
            xi = par.xi[ishock]
            xi_w = par.xi_w[ishock]
            weight = psi_w*xi_w

            # ii. next-period extra income component
            delta_plus = par.grid_delta[idelta]/(psi*par.G)
            if t == 0:
                delta_plus *= par.zeta

            # ii. next-period cash-on-hand
            for ia in range(par.Na):
                m_plus[ia] = par.R*par.grid_a[ia]/(psi*par.G) + xi
                
            # iii. next-period consumption
            if par.Ndelta > 1:
                prep = linear_interp.interp_2d_prep(par.grid_delta,delta_plus,par.Na)
            else:
                prep = linear_interp.interp_1d_prep(par.Na)
                            
            if par.Ndelta > 1:
                linear_interp.interp_2d_only_last_vec_mon(prep,par.grid_delta,par.grid_m,sol.c[t_plus],delta_plus,m_plus,c_plus)
            else:
                linear_interp.interp_1d_vec_mon(prep,par.grid_m,sol.c[t_plus,idelta],m_plus,c_plus)

            # iv. accumulate all
            for ia in range(par.Na):
                q[t,idelta,ia] += weight*par.R*par.beta*utility.marg_func(par.G*psi*c_plus[ia],par)