import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d
from consav.quadrature import log_normal_gauss_hermite

class BufferStockModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 20 # time periods
        
        # preferences
        par.beta = 0.98 # discount factor
        par.rho = 2.0 # CRRA coefficient

        # income
        par.G = 1.03 # income growth level
        par.sigma_trans = 0.1 # transitory income shock standard deviation
        par.sigma_perm = 0.1 # permanent income shock standard deviation

        # saving
        par.r = 0.02 # interest rate

        # grid
        par.m_max = 20.0 # maximum point in resource grid
        par.Nm = 50 # number of grid points in resource grid    

        par.Nxi = 5 # number of points in transitory income shock expectaion (quadrature nodes, K)
        par.Npsi = 5  # number of points in permanent income shock expectaion (quadrature nodes, J)
        # hence 25 points, we want to keep number of points low as we have to recalculate Gauss-Hermite

        # simulation
        par.seed = 9210
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim
        
        # a. asset grid
        par.m_grid = nonlinspace(0.00001,par.m_max,par.Nm,1.1) # always have a bit of resources

        # b. income shock grids
        par.xi_grid,par.xi_weight = log_normal_gauss_hermite(par.sigma_trans,par.Nxi)
        par.psi_grid,par.psi_weight = log_normal_gauss_hermite(par.sigma_perm,par.Npsi)

        # b. solution arrays
        shape = (par.T,par.Nm)
        sol.c = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # c. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.m = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.C = np.nan + np.zeros(shape)
        sim.M = np.nan + np.zeros(shape)
        sim.A = np.nan + np.zeros(shape)
        sim.P = np.nan + np.zeros(shape)
        sim.Y = np.nan + np.zeros(shape)
        
        # d. initialization
        sim.a_init = np.zeros(par.simN)
        sim.P_init = np.ones(par.simN)

        # e. random log-normal mean one income shocks
        np.random.seed(par.seed)
        sim.xi = np.exp(par.sigma_trans*np.random.normal(size=shape) - 0.5*par.sigma_trans**2)
        sim.psi = np.exp(par.sigma_perm*np.random.normal(size=shape) - 0.5*par.sigma_perm**2)


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        t = par.T-1
        sol.c[t,:] = par.m_grid
        sol.V[t,:] = self.util(sol.c[t,:])

        # c. loop backwards [note, the last element, N, in range(N) is not included in the loop due to index starting at 0]
        for t in reversed(range(par.T-1)):

            # i. loop over state varible: resources in beginning of period
            for im,resources in enumerate(par.m_grid):

                # ii. find optimal consumption at this level of resources in this period t.

                # objective function: negative since we minimize
                obj = lambda c: - self.value_of_choice(c[0],resources,t) # c[0] instead of c is done because scipy expects array

                # bounds on consumption
                lb = 0.000001 # avoid dividing with zero, no negative amount can be consumed
                ub = resources # no borrowing, max consume is ressources

                # call optimizer
                c_init = np.array(0.5*ub) # initial guess on optimal consumption
                res = minimize(obj,c_init,bounds=((lb,ub),),method='SLSQP')
                
                # store results
                sol.c[t,im] = res.x[0]
                sol.V[t,im] = -res.fun # value function
        

    def value_of_choice(self,cons,resources,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. utility from consumption
        util = self.util(cons)
        
        # c. expected continuation value from savings
        V_next = sol.V[t+1]
        assets = resources - cons
        
        # loop over income shocks 
        EV_next = 0.0 # initialize so we can execute line: EV_next += V_next_interp*par.xi_weight[i_xi]*par.psi_weight[i_psi]
        # loop thourugh combination of shocks that will be realized e.g. high transitory shock and low permanent shock
        for i_xi,xi in enumerate(par.xi_grid): 
            for i_psi,psi in enumerate(par.psi_grid):
                fac = par.G*par.psi_grid[i_psi] # normalization factor

                # interpolate next period value function for this combination of transitory and permanent income shocks
                m_next = (1.0+par.r)*assets/fac + par.xi_grid[i_xi]
                V_next_interp = interp_1d(par.m_grid,V_next,m_next) # for given level of consumption and these two draws of income shocks
                V_next_interp = (fac**(1.0-par.rho)) * V_next_interp # normalization factor

                # weight the interpolated value with the likelihood
                EV_next += V_next_interp*par.xi_weight[i_xi]*par.psi_weight[i_psi] # note we dont care about storing value but then we have to loop through all 25 nodes as expected value only makes sense after 25 loops!

        # d. return value of choice
        return util + par.beta*EV_next


    def util(self,c):
        par = self.par

        return (c)**(1.0-par.rho) / (1.0-par.rho)


    ##############
    # Simulation #
    def simulate(self):
        
        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize assets and permanent income
            sim.a[i,0] = sim.a_init[i]
            sim.P[i,0] = sim.P_init[i]

            for t in range(par.simT):
                if t<par.T: # check that simulation does not go further than solution                 

                    # iii. interpolate optimal consumption (normalized)
                    fac = par.G*sim.psi[i,t]
                    sim.m[i,t] = (1.0+par.r)*sim.a[i,t]/fac + sim.xi[i,t]
                    sim.c[i,t] = interp_1d(par.m_grid,sol.c[t],sim.m[i,t])

                    # iv. store savings and income (next-period state)
                    if t<par.simT-1:
                        sim.a[i,t+1] = sim.m[i,t] - sim.c[i,t]
                        sim.P[i,t+1] = par.G*sim.P[i,t]*sim.psi[i,t+1]

        # c. store non-normalized variables in levels
        sim.A = sim.a*sim.P
        sim.M = sim.m*sim.P
        sim.C = sim.C*sim.P
        sim.Y = sim.P*sim.psi