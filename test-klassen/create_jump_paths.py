import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class test_data():
    def __init__(self,lam=5):
        self.S   = 100  # current stock price
        self.T   = 1    # time to maturity
        self.r   = 0.02 # risk free rate
        self.m   = 0    # meean of jump size
        self.v   = 0.3  # standard deviation of jump
        self.lam = lam  # intensity of jump i.e. number of jumps per annum
        self.steps  = 10000 # time steps
        self.Npaths = 1     # number of paths to simulate
        self.sigma  = 0.2   # annaul standard deviation , for weiner process

        self.j, self.jumps = self.merton_jump_paths()

        self.jumps_x = list(np.ndarray.nonzero(self.jumps))[0]
        self.jumps_y = self.j[self.jumps_x]

    def merton_jump_paths(self):
        size=(self.steps,self.Npaths)
        dt = self.T/self.steps
        # poisson- distributed jumps
        jumps = np.random.poisson( self.lam*dt, size=size)

        poi_rv = np.multiply(jumps,
                             np.random.normal(self.m,self.v, size=size)).cumsum(axis=0)
        geo = np.cumsum(((self.r -  self.sigma**2/2 -self.lam*(self.m  + self.v**2*0.5))*dt +
                         self.sigma*np.sqrt(dt) *
                         np.random.normal(size=size)), axis=0)

        return np.exp(geo+poi_rv)*self.S, jumps

    def plot_path_jumps(self):
        plt.figure(figsize=(12,10))
        plt.plot(self.j, c= 'blue',label='time-series')
        plt.plot(self.jumps_x,self.jumps_y,"o",c='red',label='jumps')
        plt.grid(True)
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title('Jump Diffusion Process')
        plt.legend(loc='best')
        plt.show()

