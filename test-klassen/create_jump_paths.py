import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

class test_data():
    def __init__(self,lam=5):
        self.S   = 100  # current stock price
        self.T   = 1    # time to maturity
        self.r   = 0.02 # risk free rate
        self.m   = 0    # meean of jump size
        self.v   = 0.3  # standard deviation of jump
        self.lam = lam  # intensity of jump i.e. number of jumps per annum
        self.steps  = 1000 # time steps
        self.Npaths = 1     # number of paths to simulate
        self.sigma  = 0.2   # annaul standard deviation , for weiner process

        self.data, self.jumps   = self.merton_jump_paths()
        self.data_df = self.comp_data_df()

        self.returns = self.calc_returns()
        self.rv = self.calc_rv()

        self.jumps_x = list(np.ndarray.nonzero(self.jumps))[0]
        self.jumps_y = self.data[self.jumps_x]

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

    def calc_returns(self):
        returns = pd.DataFrame()
        returns['Return'] = self.data_df['0'].pct_change()
        returns = returns.dropna()
        returns.columns = ['0']
        return returns

    def calc_rv(self):
        # Muss Überprüft werden !!!!
        rv = pd.DataFrame()
        rv['RV'] = self.returns['0'].rolling(2).var()
        rv = rv.dropna()
        return rv

    def comp_data_df(self):
        return pd.DataFrame(self.data,columns=['0'])

    def plot_path_jumps(self):
        plt.figure(figsize=(12,10))
        plt.plot(self.data, c= 'blue',label='time-series')
        plt.plot(self.jumps_x,self.jumps_y,"o",c='red',label='jumps')
        plt.grid(True)
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title('Jump Diffusion Process')
        plt.legend(loc='best')
        plt.show()

    def plot_variations(self):
        fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True)
        fig.suptitle('Merkmale')
        fig.set_size_inches(12,10)

        ax1.plot(self.returns, label='Returns')
        ax2.plot(self.rv, label='Realized variation')
        ax1.grid(True)
        ax2.grid(True)
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.show()