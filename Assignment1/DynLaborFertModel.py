import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint

import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborFertModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.rho = 1/1.02 # discount factor

        par.beta_0 = 0.1 # weight on labor dis-utility (constant)
        par.beta_1 = 0.05 # additional weight on labor dis-utility (children)
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.3 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.1 # labor income tax

        # children
        par.p_birth = 0.1

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 50 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 20 # number of grid points in wealth grid    

        par.Nn = 2 # number of children

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1.1)

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # c. number of children grid
        par.n_grid = np.arange(par.Nn)

        # d. solution arrays
        shape = (par.T,par.Nn,par.Na,par.Nk)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)

        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        # g. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)

        # h. vector of wage and tax 
        par.w_vec = par.w * np.ones(par.T)
        par.tau_vec = par.tau * np.ones(par.T) # [Question 2]
        
        # i. target drop
        par.target_birth_drop = -0.10 # [Question 1]


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # i. loop over state variables: number of children, human capital and wealth in beginning of period
            for i_n,kids in enumerate(par.n_grid):
                for i_a,assets in enumerate(par.a_grid):
                    for i_k,capital in enumerate(par.k_grid):
                        idx = (t,i_n,i_a,i_k)

                        # ii. find optimal consumption and hours at this level of wealth in this period t.

                        if t==par.T-1: # last period
                            obj = lambda x: self.obj_last(x[0],assets,capital,kids)

                            constr = lambda x: self.cons_last(x[0],assets,capital)
                            nlc = NonlinearConstraint(constr, lb=0.0, ub=np.inf,keep_feasible=True)

                            # call optimizer
                            hours_min = - assets / self.wage_func(capital,t) + 1.0e-5 # minimum amout of hours that ensures positive consumption
                            hours_min = np.maximum(hours_min,2.0)
                            init_h = np.array([hours_min]) if i_a==0 else np.array([sol.h[t,i_n,i_a-1,i_k]]) # initial guess on optimal hours

                            res = minimize(obj,init_h,bounds=((0.0,np.inf),),constraints=nlc,method='trust-constr')

                            # store results
                            sol.c[idx] = self.cons_last(res.x[0],assets,capital)
                            sol.h[idx] = res.x[0]
                            sol.V[idx] = -res.fun

                        else:
                            
                            # objective function: negative since we minimize
                            obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,kids,t)  

                            # bounds on consumption 
                            lb_c = 0.000001 # avoid dividing with zero
                            ub_c = np.inf

                            # bounds on hours
                            lb_h = 0.0
                            ub_h = np.inf 

                            bounds = ((lb_c,ub_c),(lb_h,ub_h))
                
                            # call optimizer
                            init = np.array([lb_c,1.0]) if (i_n == 0 & i_a==0 & i_k==0) else res.x  # initial guess on optimal consumption and hours
                            res = minimize(obj,init,bounds=bounds,method='L-BFGS-B') 
                        
                            # store results
                            sol.c[idx] = res.x[0]
                            sol.h[idx] = res.x[1]
                            sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital):
        par = self.par

        income = self.wage_func(capital,par.T-1) * hours
        cons = assets + income
        return cons

    def obj_last(self,hours,assets,capital,kids):
        cons = self.cons_last(hours,assets,capital)
        return - self.util(cons,hours,kids)    

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,kids,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. penalty for violating bounds. 
        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0
            cons = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. utility from consumption
        util = self.util(cons,hours,kids)
        
        # d. *expected* continuation value from savings
        income = self.wage_func(capital,t) * hours
        a_next = (1.0+par.r)*(assets + income - cons)
        k_next = capital + hours

        # no birth
        kids_next = kids
        V_next = sol.V[t+1,kids_next]
        V_next_no_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        # birth
        if (kids>=(par.Nn-1)):
            # cannot have more children
            V_next_birth = V_next_no_birth

        else:
            kids_next = kids + 1
            V_next = sol.V[t+1,kids_next]
            V_next_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        EV_next = par.p_birth * V_next_birth + (1-par.p_birth)*V_next_no_birth

        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours,kids):
        par = self.par

        beta = par.beta_0 + par.beta_1*kids

        return (c)**(1.0+par.eta) / (1.0+par.eta) - beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func(self,capital,t):
        # after tax wage rate
        par = self.par

        return (1.0 - par.tau_vec[t] )* par.w_vec[t] * (1.0 + par.alpha * capital)

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.n[i,0] = sim.n_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.n[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.wage_func(sim.k[i,t],t)*sim.h[i,t]
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    birth = 0 
                    if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1))):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth
                
                
    #########################
    # Structural Estimation #    
    def moment(self,beta_1):
        """ Compute drop in labor supply along intensive margin in the event of a birth. """

        self.solve()
        self.simulate()

        # a. unpack parameters
        par, sim, sol = self.par, self.sim, self.sol
        par.beta_1 = beta_1

        # b. moment
        birth = np.zeros(sim.n.shape, dtype=np.int_)
        birth[:,1:] = (sim.n[:,1:] - sim.n[:,:-1]) > 0

        # c. time since birth
        periods = np.tile([t for t in range(par.simT)],(par.simN,1)) 
        time_of_birth = np.max(periods * birth, axis=1)
        I = time_of_birth > 0 
        time_of_birth[~I] = -1000
        time_of_birth = np.transpose(np.tile(time_of_birth,(par.simT,1)))
        time_since_birth = periods - time_of_birth

        # d. calculate the percentage change in hours from the period before birth
        hours_before = np.mean(sim.h[time_since_birth==0])
        hours_after = np.mean(sim.h[time_since_birth==-1])

        est_birth_drop = (hours_before / hours_after -1)
        
        return est_birth_drop
        
    def structural_est(self,beta_1):
        """ Estimation of beta_1. """

        # a. unpack
        par = self.par
        
        # b. objective function
        diff = self.moment(beta_1) - par.target_birth_drop
        return np.abs(diff)
        
###############
# Event Study #

def event_study_graph(model, ax = None):
    """ Plots the percentage change in hours worked by individuals in a population compared to the period before their birth.

    Parameters:
        model (Model): An instance of the Model class.
        ax (matplotlib.axes.Axes, optional): The plot axis. If not provided, a new figure and axis will be created.


    Returns:
        ax (matplotlib.axes.Axes): The plot axis.
    """
    
    # a. unpack parameters
    par, sim = model.par, model.sim
    
    # b. time since birth
    birth = np.zeros(sim.n.shape, dtype=np.int_)
    birth[:,1:] = (sim.n[:,1:] - sim.n[:,:-1]) > 0
    periods = np.tile([t for t in range(par.simT)], (par.simN, 1)) 
    time_of_birth = np.max(periods * birth, axis=1)
    I = time_of_birth > 0 
    
    # Set time of birth to -1000 for those who are never considered as a child, 
    # ensuring they don't affect later calculations
    time_of_birth[~I] = -1000
    time_of_birth = np.transpose(np.tile(time_of_birth, (par.simT, 1)))
    time_since_birth = periods - time_of_birth

    # c. calculate average outcome across time since birth
    min_time = -8
    max_time = 8
    event_grid = np.arange(min_time, max_time+1)

    event_hours = np.nan + np.zeros(event_grid.size)
    # calculate the average hours worked for each time period in event grid
    event_hours = np.array([np.mean(sim.h[time_since_birth == time]) for time in event_grid])
    
    # d. calculate the percentage change in hours from the period before birth
    event_hours_rel = (event_hours / event_hours[event_grid == -1] - 1) * 100
    
    # e. calculate plot axis
    if ax is None:
        _, ax = plt.subplots()  # create a new plot axis
    
    ax.scatter(event_grid, event_hours_rel, label=f'$\\beta_1$={round(par.beta_1, 3)}')
    ax.hlines(y=0, xmin=event_grid[0], xmax=event_grid[-1], color='gray')
    ax.vlines(x=-0.5, ymin=np.nanmin(event_hours_rel), ymax=np.nanmax(event_hours_rel), color='red')
    ax.set(xlabel='Time since birth', ylabel='Hours worked (% change)', xticks=event_grid)
    ax.legend(frameon=True)
    
    return ax

        
##########################
# Marshallian Elasticity #
def simulate_marshallian_elasticity(model, tax_increase):
    """ 
    Simulates Marshallian elasticity for all periods in a given model after a permanent increase in marginal taxes.
    
    Parameters:
        model (object): The model to simulate.
        tax_increase (float): The percent increase in taxes.
    
    Returns:
        model_increase (object): The model with 1 pct higher wage.
        ela_total, ela_child, ela_no_child (numpy array): The Marshallian elasticity from simulated tax increase.
    """
    
    # a. unpack parameters
    par, sim = model.par, model.sim
    
    # b. create a copy of the model with increased wages and solve
    model_increase = model.copy()
    model_increase.par.tau_vec[:] *= tax_increase
    model_increase.solve()
    
    # c. simulate the model and calculate elasticity
    model_increase.simulate()

    # Total elasticity
    ela_total = (model_increase.sim.h - model.sim.h) / model.sim.h

    # Allocate
    ela_child = np.full(model_increase.par.T, np.nan)
    ela_no_child = np.full(model_increase.par.T, np.nan)

    # Calculate elasticity for each group
    for t in range(model_increase.par.T):  
        without_child = sim.n[:,t] == 0
        ela_no_child[t] = np.nanmean((model_increase.sim.h[without_child,t] - model.sim.h[without_child,t]) / model.sim.h[without_child,t])

    for t in range(1, model_increase.par.T):  
        with_child = sim.n[:,t] > 0
        ela_child[t] = np.nanmean((model_increase.sim.h[with_child,t] - model.sim.h[with_child,t]) / model.sim.h[with_child,t])

    # d. print average Marshallian elasticity
    print(f'Total Average Marshallian Elasticity: {np.nanmean(ela_total):2.6f}')
    print(f'Average Marshallian Elasticity with child: {np.nanmean(ela_child):2.6f}')
    print(f'Average Marshallian Elasticity without child: {np.nanmean(ela_no_child):2.6f}')

    # e. return the average elasticity
    return model_increase, ela_total, ela_child, ela_no_child
