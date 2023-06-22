from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
import os

from EconModel import EconModelClass
from consav.grids import nonlinspace
from consav.linear_interp_2d import interp_2d

import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 12})

# global gender identifiers
woman = 1
man = 2

class DynamicSpecializationModelClass(EconModelClass):
    #########
    # Setup #
    def settings(self):
        """ fundamental settings """
        pass

    def setup(self):
        """ setup model """

        # a. unpack
        par = self.par 

        par.T = 8

        # b. preferences
        par.beta = 0.98
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon_f = 1.0
        par.epsilon_m = 1.0
        par.omega = 0.5 
        par.omega_child = -0.2 # extension
        par.p_birth = 0.1 # extension

        # c. household production
        par.alpha = 0.7
        par.sigma = 0.3

        # d. wages and income
        par.wage_const_f = 0.0
        par.wage_humcap_f = 0.1
        par.wage_const_m = 0.0
        par.wage_humcap_m = 0.1

        par.X = 0.0 # unearned income

        par.delta = 0.1 # human capital depreciation
        par.sigma_w = 0.0 # measurement error variance

        # e. grids
        par.num_K = 10
        par.max_K = 15.0
        par.Nn = 2 # number of children + 1

        # f. simulation
        par.simT = par.T
        par.simN = 10_000

    def allocate(self):
        # a. unpack
        par = self.par 
        sol = self.sol 
        sim = self.sim

        # b. setup grids for human capital
        par.grid_K = nonlinspace(0.0,par.max_K,par.num_K,1.1)
        
        # c. number of children grid
        par.n_grid = np.arange(par.Nn) # extension

        # d. solution
        shape = (par.T,par.Nn,par.num_K,par.num_K) # extension
        sol.V = np.nan + np.zeros(shape)
        sol.labor_f = np.nan + np.zeros(shape)
        sol.home_f = np.nan + np.zeros(shape)
        sol.labor_m = np.nan + np.zeros(shape)
        sol.home_m = np.nan + np.zeros(shape)

        # e. simulation
        shape = (par.simN,par.simT)
        sim.labor_f = np.nan + np.zeros(shape)
        sim.home_f = np.nan + np.zeros(shape)
        sim.labor_m = np.nan + np.zeros(shape)
        sim.home_m = np.nan + np.zeros(shape)
        sim.wage_f = np.nan + np.zeros(shape)
        sim.wage_m = np.nan + np.zeros(shape)
        sim.kf = np.nan + np.zeros(shape)
        sim.km = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_) # extension
        
        # initial values
        np.random.seed(2023)
        sim.init_kf = np.random.uniform(0,3,size=par.simN) 
        sim.init_km = np.random.uniform(0,3,size=par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_) # extension
        sim.draws_uniform = np.random.uniform(size=shape) # extension
    
    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. loop backwards (over all periods)
        for t in reversed(range(par.T)):
            if t==par.T-1:
                V_next = None
            else:
                V_next = sol.V[t+1]

            # c. loop over state variables: presence of child and human capital for each household member
            for i_n,kids in enumerate(par.n_grid):
                    
                for i_kf,capital_f in enumerate(par.grid_K):
                    for i_km,capital_m in enumerate(par.grid_K):
                        idx = (t,i_n,i_kf,i_km)
                        idx_last = (t+1,i_n,i_kf,i_km)


                        # d. find optimal time-allocation at this level of human capital.
                        obj = lambda x: self.value_of_choice(x,kids,capital_f,capital_m,V_next)  

                        bounds = [(0,24) for i in range(4)]

                        # e. initial values: use solution from future, if possible
                        if t<par.T-1:
                            init = np.array([getattr(sol,name)[idx_last] for name in ('labor_f','home_f','labor_m','home_m')])

                        else:
                            if i_kf>0 & i_km>0:
                                init = res.x
                            else:
                                init = np.ones(4)

                        # f. estimate and store results
                        res = minimize(obj,init,bounds=bounds,method='L-BFGS-B') 

                        sol.labor_f[idx] = res.x[0]
                        sol.home_f[idx] = res.x[1]
                        sol.labor_m[idx] = res.x[2]
                        sol.home_m[idx] = res.x[3]
                        sol.V[idx] = -res.fun
    
    def clip_and_penalty(self,labor_f,home_f,labor_m,home_m):

        # penalty for bounds
        penalty = 0.0
        if labor_f>24.0:
            penalty += (labor_f-24)*1000.0
            labor_f = 24.0
        if home_f>24.0:
            penalty += (home_f-24)*1000.0
            home_f = 24.0
        if labor_m>24.0:
            penalty += (labor_m-24)*1000.0
            labor_m = 24.0
        if home_m>24.0:
            penalty += (home_m-24)*1000.0
            home_m = 24.0

        # penalty for inequality constraint
        if (labor_f+home_f)>24:
            penalty += (labor_f+home_f-24)*1000.0
            labor_f = 24 - home_f
        if (labor_m+home_m)>24:
            penalty += (labor_m+home_m-24)*1000.0
            labor_m = 24 - home_m

        return labor_f,home_f,labor_m,home_m,penalty

    def value_of_choice(self,x,kids,capital_f,capital_m,V_next=None):

        # a. unpack
        par = self.par
        sol = self.sol
        
        labor_f,home_f,labor_m,home_m = x

        # b. penalty for bounds and inequality constraint
        labor_f,home_f,labor_m,home_m,penalty = self.clip_and_penalty(labor_f,home_f,labor_m,home_m)

        # c. current utility
        utility = self.util(kids,labor_f,home_f,labor_m,home_m,capital_f,capital_m)
        
        # c. continuation value
        # Extension
        if V_next is not None:
            kf_next = self.human_capital_next(capital_f,labor_f)
            km_next = self.human_capital_next(capital_m,labor_m)
            
            # no birth 
            kids_next = kids
            V_interp_no_birth = interp_2d(par.grid_K,par.grid_K,V_next[kids_next],kf_next,km_next)
            
            # birth 
            if (kids>=(par.Nn-1)):
            # cannot have more children
                V_interp_birth = V_interp_no_birth
                
            else: # birth
                kids_next = kids + 1
                V_interp_birth = interp_2d(par.grid_K,par.grid_K,V_next[kids_next],kf_next,km_next)
                
            V_interp = par.p_birth * V_interp_birth + (1-par.p_birth)*V_interp_no_birth
            
        else:
            V_interp = 0.0
            
        # d. return negative value of choice
        return -(utility + par.beta*V_interp) + penalty

    def util(self,kids,labor_f,home_f,labor_m,home_m,capital_f,capital_m):
        # a. unpack
        par = self.par

        # b. labor income and consumption expenditures
        wage_f = self.wage_func(capital_f,woman)
        wage_m = self.wage_func(capital_m,man)

        C = wage_f*labor_f + wage_m*labor_m + par.X

        # c. home production
        F_term = par.alpha*np.fmax(home_f,1e-8)**((par.sigma-1)/par.sigma)
        M_term = (1-par.alpha)*np.fmax(home_m,1e-8)**((par.sigma-1)/par.sigma)
        H = np.fmax(M_term + F_term,1e-8)**(par.sigma/(par.sigma-1))

        # d. composite good
        Omega = par.omega + par.omega_child*kids # extension
        Q = (C**Omega)  * (H**(1.0-Omega)) 

        # e. utility of consumption
        util_Q = np.fmax(Q,1e-8)**(1.0-par.rho) / (1.0-par.rho)

        # f. dis-utility from work
        T_f = labor_f + home_f
        T_m = labor_m + home_m
        power_f = 1+1/par.epsilon_f
        power_m = 1+1/par.epsilon_m

        util_T = par.nu * (T_f**power_f / power_f + T_m**power_m / power_m)
        
        # g. return total utility
        return util_Q - util_T

    def wage_func(self,capital,gender):
        # a. unpack
        par = self.par

        # b. determine relevant parameters
        const = par.wage_const_f
        humcap = par.wage_humcap_f
        if gender==man:
            const = par.wage_const_m
            humcap = par.wage_humcap_m

        # c. return wage rate
        return np.exp(const + humcap*capital)

    def human_capital_next(self,capital,labor):
        return (1.0-self.par.delta)*capital + labor/24.0
    
    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # c. initialize states
            sim.kf[i,0] = sim.init_kf[i]
            sim.km[i,0] = sim.init_km[i]
            sim.n[i,0] = sim.n_init[i]

            for t in range(par.simT):

                # d. interpolate optimal allocation
                idx_sol = (t,sim.n[i,t])
                sim.labor_f[i,t] = interp_2d(par.grid_K,par.grid_K,sol.labor_f[idx_sol],sim.kf[i,t],sim.km[i,t])
                sim.home_f[i,t] = interp_2d(par.grid_K,par.grid_K,sol.home_f[idx_sol],sim.kf[i,t],sim.km[i,t])
                sim.labor_m[i,t] = interp_2d(par.grid_K,par.grid_K,sol.labor_m[idx_sol],sim.kf[i,t],sim.km[i,t])
                sim.home_m[i,t] = interp_2d(par.grid_K,par.grid_K,sol.home_m[idx_sol],sim.kf[i,t],sim.km[i,t])

                # e. store wage (+ measurement error)
                epsilon_w_f = np.random.normal(0, par.sigma_w)
                epsilon_w_m = np.random.normal(0, par.sigma_w)
                sim.wage_f[i,t] = self.wage_func(sim.kf[i,t],woman) + epsilon_w_f
                sim.wage_m[i,t] = self.wage_func(sim.km[i,t],man) + epsilon_w_m

                # f. store next-period states
                if t<par.simT-1:
                    sim.kf[i,t+1] = self.human_capital_next(sim.kf[i,t],sim.labor_f[i,t])
                    sim.km[i,t+1] = self.human_capital_next(sim.km[i,t],sim.labor_m[i,t])
                
                    # evaluate whether a child arrives
                    birth = 0 
                    if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1))):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth

        # g. specialization index
        sim.specialization = sim.home_f/(sim.home_f+sim.home_m)-sim.labor_f/(sim.labor_f+sim.labor_m)
        sim.time_since_birth = self.get_birth_time() # structural estimation
        sim.constant = self.regress()[0] # structural estimation
        sim.slope = self.regress()[1] # structural estimation

    ##############                
    # Estimation #                     
    def regress(self):
        """ run regression """
        # a. unpack
        sim = self.sim

        # b. run regression
        if (np.min(sim.wage_f)>0.0) & (np.min(sim.wage_m)>0.0) & (np.min(sim.home_f)>0.0) & (np.min(sim.home_m)>0.0):
            x = np.log(sim.wage_f/sim.wage_m).ravel()
            y = np.log(sim.home_f/sim.home_m).ravel()
            A = np.vstack([np.ones(x.size),x]).T
            constant,slope = np.linalg.lstsq(A,y,rcond=None)[0]
        else:
            constant = slope = np.nan
        
        # c. return constant and slope parameters
        return constant,slope

    def plot(self, add_regression=True, ax=None):
        # a. unpack
        sim = self.sim

        # b. log relative wage of women 
        rel_wage = np.log(sim.wage_f/sim.wage_m)
        rel_wage_grid = np.linspace(np.min(rel_wage.ravel()),np.max(rel_wage.ravel()),2)

        # c. log relative home production 
        rel_home = np.log(sim.home_f/sim.home_m)

        # d. plot relationship 
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(rel_wage, rel_home, label='simulated scatter')
        ax.plot(rel_wage_grid, 0.4-0.1*rel_wage_grid, label='empirical target', color='orange', linewidth=2)
        if add_regression:
            constant, slope = self.regress()
            ax.plot(rel_wage_grid, constant+slope*rel_wage_grid, label='simulated relationship', color='red', linewidth=2)

        ax.set(xlabel='$log(w_f/w_m)$', ylabel='$log(h_f/h_m)$')
        ax.legend(fontsize=10)  
        ax.xaxis.label.set_size(10)  
        ax.yaxis.label.set_size(10)  
        plt.tight_layout()
        return ax
        
    def get_birth_time(self):

        # a. unpack
        par = self.par
        sim = self.sim

        # b. Calculate birth times
        birth = np.zeros(sim.n.shape,dtype=np.int_)
        birth[:,1:] = (sim.n[:,1:] - sim.n[:,:-1]) > 0

        # c. Calculate time since birth
        periods = np.tile([t for t in range(par.simT)],(par.simN,1))
        time_of_birth = np.max(periods * birth, axis=1)

        I = time_of_birth>0
        time_of_birth[~I] = 1000 # never has a child
        time_of_birth = np.transpose(np.tile(time_of_birth , (par.simT,1)))

        time_since_birth = periods - time_of_birth

        return time_since_birth

        
##############
# Event Study#
def specilization_graph(model, relative_change=False, ax=None):
    
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
    min_time = -7
    max_time = 6
    event_grid = np.arange(min_time, max_time+1)

    # calculate the average specialization index for each time period in event grid
    event_specialization = np.array([np.mean(sim.specialization[time_since_birth == time]) for time in event_grid])
    
    # d. calculate the relative change in specialization from the period before birth
    event_specialization_rel = (event_specialization / event_specialization[event_grid == -1] - 1) * 100
    
    # e. calculate plot axis
    if ax is None:
        _, ax = plt.subplots()  # create a new plot axis
    
    if relative_change:
        ax.scatter(event_grid, event_specialization_rel, label = 'Extended model')
        ax.hlines(y=0, xmin=event_grid[0], xmax=event_grid[-1], color='gray', linestyle='--')
        ax.vlines(x=-0.5, ymin=np.nanmin(event_specialization_rel), ymax=np.nanmax(event_specialization_rel), color='red')
    
    else:
        ax.plot(event_grid, event_specialization, label = 'Extended model')

    ax.legend()
    
    return ax