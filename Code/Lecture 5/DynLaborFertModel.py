import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn off annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborFertModelClass(EconModelClass):

    def settings(self):
        """ Fundamental settings. """

        pass

    def setup(self):
        """ Set baseline parameters. """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.rho = 0.98 # discount factor

        par.beta_0 = 0.1 # weight on labor dis-utility (constant)
        par.beta_1 = 0.05 # additional weight on labor dis-utility (children)
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.1 # human capital accumulation / mapping work experience to human capital
        par.w = 1.0 # wage base level / skill endowment
        par.tau = 0.1 # labor income tax
        
        # spouse
        par.y = 1.0 # to remove spouse set equal to 0
        par.p_spouse = 0.8 # probability of a spouse being present in a given period

        # children
        par.p_birth = 0.1
        par.theta = 0.05 # childcare costs

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 50 #70 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 20 #30 # number of grid points in wealth grid    

        par.Nn = 2 # number of children + 1

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals


    def allocate(self):
        """ allocate model """

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # b. asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1.1)

        # c. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # d. number of children grid (new vis-á-vis lecture 4)
        par.n_grid = np.arange(par.Nn) # [0, 1]

        # e. solution arrays for given period, number of children, wealth, and human capital
        shape = (par.T,par.Nn,par.Na,par.Nk) # [10, 2, 50, 20], we include par.Nn in the second position to easier interpolate par.Na,par.Nk
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # f. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)

        # g. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        # h. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)

        # i. vector of wages and presence of spouse in a given period
        par.w_vec = par.w * np.ones(par.T) # w_vec allows us to change elasticity at different point in time
        par.spouse_vec = np.random.binomial(1, par.p_spouse, size=par.T)  # draws a spouse presence variable for each period

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
            for i_n, kids in enumerate(par.n_grid):
                for i_a, assets in enumerate(par.a_grid):
                    for i_k, capital in enumerate(par.k_grid):
                        idx = (t, i_n, i_a, i_k)

                        # ii. find optimal consumption and hours at this level of wealth in this period t.
                        if t == par.T-1:  # last period
                            obj = lambda x: self.obj_last(x[0], assets, capital, kids) # added kids argument

                            constr = lambda x: self.cons_last(x[0], assets, capital, kids) # added kids argument
                            nlc = NonlinearConstraint(constr, lb=0.0, ub=np.inf, keep_feasible=True)

                            # call optimizer
                            childcare_cost = par.theta * kids
                            # minimum amount of hours that ensures positive consumption
                            hours_min = (childcare_cost - assets) / self.wage_func(capital, t) + 1.0e-5 # updated such that childcare costs doesn't violate initial guess that ensure positive consumption
                            hours_min = np.maximum(hours_min, 2.0)
                            init_h = np.array([hours_min]) if i_a == 0 else np.array([sol.h[t, i_n, i_a-1, i_k]])  # initial guess on optimal hours

                            res = minimize(obj, init_h, bounds=((0.0, np.inf),), constraints=nlc, method='trust-constr')

                            # store results
                            sol.c[idx] = self.cons_last(res.x[0], assets, capital, kids) # added kids argument
                            sol.h[idx] = res.x[0]
                            sol.V[idx] = -res.fun

                        else:

                            # objective function: negative since we minimize
                            obj = lambda x: - self.value_of_choice(x[0], x[1], assets, capital, kids, t) # added kids argument

                            # bounds on consumption
                            lb_c = 0.000001  # avoid dividing with zero
                            ub_c = np.inf

                            # bounds on hours
                            lb_h = 0.0
                            ub_h = np.inf

                            bounds = ((lb_c, ub_c), (lb_h, ub_h))

                            # call optimizer
                            init = np.array([lb_c, 1.0]) if (i_n == 0 & i_a == 0 & i_k == 0) else res.x # initial guess on optimal consumption and hours else use result from previous loop
                            res = minimize(obj, init, bounds=bounds, method='L-BFGS-B')

                            # store results
                            sol.c[idx] = res.x[0]
                            sol.h[idx] = res.x[1]
                            sol.V[idx] = -res.fun

    # last period
    def cons_last(self, hours, assets, capital, kids):
        """
        Calculate the consumption in the last period.

        Args:
            hours (float): Hours of work.
            assets (float): Assets at the beginning of the period.
            capital (float): Human capital at the beginning of the period.
            kids (int): Number of children.

        Returns:
            float: Consumption in the last period.
        """
        # unpack
        par = self.par
        
        income = self.wage_func(capital, par.T-1) * hours # household's income process
        childcare_cost = par.theta * kids # added childcare cost term
        cons = assets + income - childcare_cost # ensures all is consumed in last period as no bequest motive
        
        return cons

    # last period
    def obj_last(self, hours, assets, capital, kids): # equivalent to math
        """ Objective function for the last period.
    
        Args:
            hours (float): Hours of work.
            assets (float): Assets at the beginning of the period.
            capital (float): Human capital at the beginning of the period.
            kids (int): Number of children.

        Returns:
            float: Negative utility in the last period (remember, we are minimizing)
        """
        
        cons = self.cons_last(hours, assets, capital, kids) # added kids argument
        return - self.util(cons, hours, kids)

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,kids,t):
        """ Calculate the value of a consumption and hours choice for a given state.

        Args:
            cons (float): Consumption.
            hours (float): Hours of work.
            assets (float): Assets at the beginning of the period.
            capital (float): Human capital at the beginning of the period.
            kids (int): Number of children (max 1).
            t (int): Time period.

        Returns:
            float: Value of the choice.
        """
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
        
        # d. *expected* continuation value from savings | value of next period
        income = self.wage_func(capital, t) * hours
        childcare_cost = par.theta * kids # added childcare cost term
        a_next = (1.0 + par.r) * (assets + income - cons - childcare_cost) # transition equation
        k_next = capital + hours # transition equation

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


    def util(self, c, hours, kids):
        """ Calculate the utility function.

        Args:
            c (float): Consumption.
            hours (float): Hours of work.
            kids (int): Number of children.

        Returns:
            float: Utility value.
        """
        # unpack
        par = self.par

        beta = par.beta_0 + par.beta_1 * kids # children imposes a larger disutility on work reflecting amenity value with regard to children
        return (c) ** (1.0 + par.eta) / (1.0 + par.eta) - beta * (hours) ** (1.0 + par.gamma) / (1.0 + par.gamma)


    def wage_func(self, capital, t):
        """ Calculate the after-tax wage rate.

        Args:
            capital (float): Human capital at the beginning of the period.
            t (int): Time period.

        Returns:
            float: After-tax wage rate.
        """   
        
        # unpack
        par = self.par

        # add spouse's age dependent after-tax income
        spouse_income = (1-par.tau)*(0.1 + 0.01 * t) * par.y * par.spouse_vec[t]  # multiply by spouse presence variable

        return (1.0 - par.tau) * par.w_vec[t] * (1.0 + par.alpha * capital) + spouse_income # Mincer type endogenous wage + spouse' income

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

                    # evaluate whether a child arrives (spousal presence needed)
                    birth = 0 
                    if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1)) & (par.spouse_vec[t] == 1)):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth

def simulate_marshallian_elasticity(model, increase):
    """ Simulates Marshall elasticity for all periods in a given model after a permanent increase in wages.
    
    Parameters:
        model (object): The model to simulate.
        (wage) increase (float): The percent increase in wages.
    
    Returns:
        model_increase (object): The model with 1 pct higher wage.
        ela_Mi (numpy array): The Marshallian elasticity from simulated wage increase.
    """
    # a. unpack parameters
    par, sim = model.par, model.sim
    
    # b. create a copy of the model with increased wages and solve
    model_increase = model.copy()
    model_increase.par.w_vec[:] *= increase
    model_increase.solve()
    
    # c. simulate the model and calculate elasticity
    model_increase.simulate()
    ela_Mi = (model_increase.sim.h - model.sim.h)/model.sim.h*100

    # d. print average marshall elasticity
    print(f'Average Marshallian Elasticity: {np.mean(ela_Mi):2.3f}')
    
    # e. return the average elasticity
    return model_increase, ela_Mi

import matplotlib.pyplot as plt
def event_study_graph(model, ax = None):
    """ Plots the average number of hours worked by individuals in a population relative to the period before their birth.

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
    time_of_birth[~I] = -1000 # ensure they are never considered as a child in later calculations
    time_of_birth = np.transpose(np.tile(time_of_birth, (par.simT, 1)))
    time_since_birth = periods - time_of_birth

    # c. calculate average outcome across time since birth
    min_time = -8
    max_time = 8
    event_grid = np.arange(min_time, max_time+1)

    event_hours = np.nan + np.zeros(event_grid.size)
    event_hours = np.array([np.mean(sim.h[time_since_birth == time]) for time in event_grid]) # changed for loop to list comprehension
    
    # d. relative to period before birth
    event_hours_rel = event_hours - event_hours[event_grid==-1]
    
    # e. calculate plot axis
    if ax is None:
        _, ax = plt.subplots()  # create a new plot axis
    
    ax.scatter(event_grid, event_hours_rel, label=f'$\\beta_1$={round(par.beta_1, 3)}')
    ax.hlines(y=0, xmin=event_grid[0], xmax=event_grid[-1], color='gray')
    ax.vlines(x=-0.5, ymin=np.nanmin(event_hours_rel), ymax=np.nanmax(event_hours_rel), color='red')
    ax.set(xlabel='Time since birth', ylabel='Hours worked (rel. to -1)', xticks=event_grid)
    ax.legend(frameon=True)
    
    return ax