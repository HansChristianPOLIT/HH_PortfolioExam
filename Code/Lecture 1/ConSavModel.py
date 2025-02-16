import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d


class ConSavModelClass(EconModelClass):
# creating the general framework 

    # in .settings() you must specify some variable lists such as namespaces, blocks etc.
    def settings(self):
        """ fundamental settings """
    

        pass

    # in .setup() you choose parameters
    def setup(self):
        """ set baseline parameters """
    
        # unpack parameters to eaze notation
        par = self.par

        par.T = 20 # time periods
        
        # preferences
        par.beta = 0.98 # discount factor
        par.rho = 2.0 # CRRA coefficient

        # income
        par.y = 1.0 # income level, constant income across period

        # saving
        par.r = 0.02 # interest rate

        # grid
        par.a_max = 30.0 # maximum point in wealth grid
        par.Na = 200 # number of grid points in wealth grid      

        # simulation
        par.simT = par.T # number of periods simulated
        par.simN = 1_000 # number of individuals

    # in .allocate() all variables are automatically allocated
    def allocate(self):
        """ allocate model (grid-like structure that holds data) """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim
        
        # a. asset grid
        par.a_grid = nonlinspace(0.0,par.a_max,par.Na,1.1) # input `1.1` --> determining unequalness of spacing. Higher value more concentration in bottom of grid.

        # b. income
        par.yt = par.y * np.ones(par.T) # fill vector with income value 

        # c. solution arrays
        shape = (par.T,par.Na) # 20x200
        sol.c = np.nan + np.zeros(shape) # fill 2d matrix with value np.nan (think of np.nan as a placeholder!)
        sol.V = np.nan + np.zeros(shape) # fill 2d matrix with value np.nan 

        # d. simulation arrays
        shape = (par.simN,par.simT) # 1000x20
        sim.c = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)

        # e. initialization
        sim.a_init = np.zeros(par.simN) # start people with some amount of wealth (here zero wealth)


    ############
    # we solve backwards using BI, but simulate forward!
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period and store
        #dimension 1x200
        t = par.T-1 # terminal period
        sol.c[t,:] = par.a_grid + par.yt[t]
        sol.V[t,:] = self.util(sol.c[t,:]) # input optimal consumption vector in utility function

        # c. loop backwards [note, the last element, N, in range(N) is not included in the loop due to index starting at 0]
        for t in reversed(range(par.T-1)):

            # i. loop over state varible: wealth in beginning of period
            for ia,assets in enumerate(par.a_grid): # ia is index associated with particular value, asset, of state variable, wealth

                # ii. find optimal consumption at this level of wealth in this period t.
                # objective function (value of choice): negative since we minimize
                obj = lambda c: - self.value_of_choice(c[0],assets,t) # is because minimizes taskes array as input so we use c[0]

                # bounds on consumption
                lb = 0.000001 # avoid dividing with zero
                ub = assets + par.yt[t] # not allow to borrow, c<a+y

                # call optimizer
                c_init = np.array(0.5*ub) # initial guess on optimal consumption is half of income+wealth
                res = minimize(obj,c_init,bounds=((lb,ub),),method='SLSQP')
                
                # store results
                sol.c[t,ia] = res.x[0] # we only maximize over one element
                sol.V[t,ia] = -res.fun 
        

    def value_of_choice(self,cons,assets,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. utility from consumption
        util = self.util(cons)
        
        # c. continuation value from savings
        V_next = sol.V[t+1]
        a_next = (1.0+par.r)*(assets + par.yt[t] - cons)
        V_next_interp = interp_1d(par.a_grid,V_next,a_next) # interpolate values not on grid

        # d. return value of choice
        return util + par.beta*V_next_interp


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

            # i. initialize assets
            sim.a[i,0] = sim.a_init[i] # zero initial wealth

            for t in range(par.simT):
                if t<par.T: # check that simulation does not go further than solution

                    # ii. interpolate optimal consumption
                    sim.c[i,t] = interp_1d(par.a_grid,sol.c[t],sim.a[i,t]) # e.g. sol.c[t] is optimal value at time t and for all values of a

                    # iii. store savings (next-period state)
                    if t<par.simT-1:
                        sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + par.yt[t] - sim.c[i,t])


