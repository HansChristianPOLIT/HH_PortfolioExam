import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint

import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborFertModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings (empty). """

        pass

    def setup(self):
        """ set baseline parameters. """

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
        par.theta = 0.05

        # spouse
        par.p_spouse = 1.0 
        par.spouse_base = 0.1
        par.spouse_slope = 0.01
        par.y = 1.0
        par.spouse_rand = 1.0
        
        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 50 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 20 # number of grid points in wealth grid    

        par.Nn = 2 # number of children + 1
        par.Ns = 2 # number of spouses + 1

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals


    def allocate(self):
        """ allocate model. """

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
        
        # d. spouse grid
        par.s_grid = np.arange(par.Ns)

        shape = (par.T,par.Ns,par.Nn,par.Na,par.Nk)
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
        sim.s = np.zeros(shape,dtype=np.int_)

        # f. draws used to simulate child and spouse arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        # g. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)
        sim.s_init = np.random.choice(par.s_grid, p=[1-par.p_spouse,par.p_spouse], size=par.simN)

        # h. vector of wages. Used for simulating elasticities
        par.w_vec = par.w * np.ones(par.T)
        par.tau_vec = par.tau * np.ones(par.T)


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
            for i_s,spouse in enumerate(par.s_grid):
                for i_n,kids in enumerate(par.n_grid):
                    for i_a,assets in enumerate(par.a_grid):
                        for i_k,capital in enumerate(par.k_grid):
                            idx = (t,i_s,i_n,i_a,i_k)

                            # ii. find optimal consumption and hours at this level of wealth in this period t.

                            if t==par.T-1: # last period
                                obj = lambda x: self.obj_last(x[0],assets,capital,kids,spouse,t)

                                constr = lambda x: self.cons_last(x[0],assets,capital,kids,spouse,t)
                                nlc = NonlinearConstraint(constr, lb=0.0, ub=np.inf,keep_feasible=True)

                                # call optimizer
                                hours_min = - (assets + (par.spouse_base + par.spouse_slope * t) * par.y * (spouse > 0) - par.theta*(kids > 0)) / self.wage_func(capital,t) + 1.0e-5 # minimum amout of hours that ensures positive consumption
                                hours_min = np.maximum(hours_min,2.0)
                                init_h = np.array([hours_min]) if i_a==0 else np.array([sol.h[t,i_s,i_n,i_a-1,i_k]]) # initial guess on optimal hours

                                res = minimize(obj,init_h,bounds=((0.0,np.inf),),constraints=nlc,method='trust-constr')

                                # store results
                                sol.c[idx] = self.cons_last(res.x[0],assets,capital,kids,spouse,t)
                                sol.h[idx] = res.x[0]
                                sol.V[idx] = -res.fun

                            else:
                                
                                # objective function: negative since we minimize
                                obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,kids,spouse,t)  

                                # bounds on consumption 
                                lb_c = 0.000001 # avoid dividing with zero
                                ub_c = np.inf

                                # bounds on hours
                                lb_h = 0.0
                                ub_h = np.inf 

                                bounds = ((lb_c,ub_c),(lb_h,ub_h))
                    
                                # call optimizer
                                init = np.array([lb_c,1.0]) if (i_s == 0 & i_n == 0 & i_a==0 & i_k==0) else res.x  # initial guess on optimal consumption and hours
                                res = minimize(obj,init,bounds=bounds,method='L-BFGS-B') 
                            
                                # store results
                                sol.c[idx] = res.x[0]
                                sol.h[idx] = res.x[1]
                                sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital,kids,spouse,t):
        """
        The consumption in the last period, "comsume the rest".

        :param hours: Number of hours worked
        :param assets: Amount of assets
        :param capital: Amount of capital
        :param kids: Number of kids
        :param spouse: Income from spouse
        :param t: Time period
        :return: Consumption in the last period
        """
        
        # calculate household income
        income = self.hh_income(capital,hours,kids,spouse,t) 
        cons = assets + income
        return cons

    def obj_last(self,hours,assets,capital,kids,spouse,t):
        """
        The objective value in the last period.

        :param hours: Number of hours worked
        :param assets: Amount of assets
        :param capital: Amount of capital
        :param kids: Number of kids
        :param spouse: Income from spouse
        :param t: Time period
        :return: Objective value in the last period
        """
        
        # calculate consumption in the last period
        cons = self.cons_last(hours,assets,capital,kids,spouse,t)
        # calculate and return the negative of the utility of consumption, hours worked, and number of kids (to be maximized)
        return - self.util(cons,hours,kids)   

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,kids,spouse,t):
        """
        Value of a given choice for a household.

        Args:
        cons (float): Consumption
        hours (float): Labor hours
        assets (float): Assets
        capital (float): Capital
        kids (int): Number of children
        spouse (bool): Spouse presence indicator
        t (int): Current period

        Returns:
        float: The value of the given choice.
        """

        # a. unpack model parameters and solution variables
        par = self.par
        sol = self.sol

        # b. Apply penalty for violating bounds on consumption and hours 
        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0  # Apply penalty for negative consumption
            cons = 1.0e-5  # Set minimum possible consumption value
        if hours < 0.0:
            penalty += hours*1_000.0  # Apply penalty for negative labor hours
            hours = 0.0  # Set minimum possible labor hour value

        # c. Calculate utility from consumption and labor hours
        util = self.util(cons,hours,kids)

        # d. Calculate the expected continuation value from savings
        # Determine income based on assets, hours worked, kids, spouse, and period
        income = self.hh_income(capital,hours,kids,spouse,t)  
        a_next = (1.0+par.r)*(assets + income - cons)  # Calculate future assets
        k_next = capital + hours  # Calculate future capital

        # LOOP CODE
        if par.spouse_rand == 1:  # Check spouse condition
            # Initialize expected future value
            EV_next = 0.0 
            # Conditional statement to handle possibility of birth and existence of spouse
            num_birth = 2 if kids<par.Nn-1 and spouse == 1 else 1
            # Probabilities for child birth
            probs_n = [1-par.p_birth,par.p_birth]
            # Set iteration parameter for spouse
            num_spouse = 2
            # Probabilities for spouse presence in next period
            probs_s = [1-par.p_spouse,par.p_spouse]

            # Loop over possible states for spouse and birth
            for s_next in range(num_spouse):
                p_s_next = probs_s[s_next]  # Probability of next spouse state
                for birth in range(num_birth):
                    # Probability of next birth state
                    p_n_next = probs_n[birth] if num_birth > 1 else 1.0
                    n_next = kids + birth  # Children count in next period
                    # Calculate value in the next period
                    V_next = sol.V[t+1,s_next,n_next]
                    # Interpolate value for the next period
                    V_next_interp = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)
                    # Calculate expected future value
                    EV_next = EV_next + p_s_next * p_n_next * V_next_interp 

        else:  # If spouse is not present
            EV_next = 0.0
            num_birth = 2 if kids<par.Nn-1 else 1  # Number of possible births
            probs = [1-par.p_birth,par.p_birth]  # Probabilities for child birth

            for birth in range(num_birth):
                # Probability of next birth state
                p_n_next = probs[birth] if num_birth > 1 else 1.0
                n_next = kids + birth  # Children count in next period
                # Calculate value in the next period assuming no spouse
                V_next = sol.V[t+1,0,n_next]
                # Interpolate value for the next period
                V_next_interp = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)
                # Calculate expected future value
                EV_next = EV_next + p_n_next * V_next_interp 

        # e. Return total value + penalty
        return util + par.rho * EV_next + penalty


    def util(self,c,hours,kids):
        """
        Utility function.

        :param c: Consumption amount
        :param hours: Number of hours worked
        :param kids: Number of kids
        :return: Utility value
        """
        
        # unpack 
        par = self.par

        # calculate the value of beta based on the number of kids
        beta = par.beta_0 + par.beta_1*kids

        # calculate and return the utility value
        return (c)**(1.0+par.eta) / (1.0+par.eta) - beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func(self,capital,t):
        """
        After-tax wage rate based on the capital and time period.

        :param capital: Amount of capital
        :param t: Time period
        :return: After-tax wage rate
        """
        
        # unpack
        par = self.par

        # calculate and return the after-tax wage rate
        return (1.0 - par.tau_vec[t])* par.w_vec[t] * (1.0 + par.alpha * capital)
    
    def hh_income(self,capital,hours,kids,spouse,t):
        """
        Total household income based on the capital, hours worked, number of kids,
        spouse income, and time period.

        :param capital: Amount of capital
        :param hours: Number of hours worked
        :param kids: Number of kids
        :param spouse: Income from spouse
        :param t: Time period
        :return: Household income
        """
        
        # unpack
        par = self.par 
        
        # calculate income from the spouse
        spouse_income = par.y * (par.spouse_base + par.spouse_slope * t) * (spouse > 0) 

        # calculate the cost of childcare
        childcare_costs = par.theta * (kids > 0) 

        # calculate and return the total household income 
        income = self.wage_func(capital,t) * hours + spouse_income - childcare_costs

        return income

    ##############
    # Simulation #
    def simulate(self):
        """
        Simulate forwards the model for each household and time period, using the
        optimal consumption and hours decisions from the solution.

        Results are stored in the `sim` attribute of the class.
        """

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.s[i,0] = sim.s_init[i]
            sim.n[i,0] = sim.n_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.s[i,t],sim.n[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.hh_income(sim.k[i,t],sim.h[i,t],sim.s[i,t],sim.n[i,t],t)
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    if par.spouse_rand == 1: # Stochastic spouse model 
                        birth = 0 
                        if ((sim.draws_uniform[i,t] <= par.p_birth) & (spouse == 1) & (sim.n[i,t]<(par.Nn-1))):
                            birth = 1
                        sim.n[i,t+1] = sim.n[i,t] + birth
                        
                        spouse = 0 
                        if ((sim.draws_uniform[i,t] <= par.p_spouse)):
                            spouse = 1 
                        sim.s[i,t+1] = spouse 
                        

                    else: # Baseline model
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
