import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

sns.set_style('whitegrid')


class test_data():
    def __init__(self, lam=5):
        self.S = 100  # current stock price
        self.T = 1  # time to maturity
        self.r = 0.02  # risk free rate
        self.m = 0  # mean of jump size
        self.v = 0.3  # standard deviation of jump
        self.lam = lam  # intensity of jump i.e. number of jumps per annum
        self.steps = 1000  # time steps
        self.Npaths = 1  # number of paths to simulate
        self.sigma = 0.2  # annaul standard deviation , for weiner process

        self.data, self.jumps = self.merton_jump_paths()
        self.data_df = self.comp_data_df()

        self.returns = self.calc_returns()
        self.rv = self.calc_rv()

        self.jumps_x = list(np.ndarray.nonzero(self.jumps))[0]
        self.jumps_y = self.data[self.jumps_x]

    def merton_jump_paths(self):
        size = (self.steps, self.Npaths)
        dt = self.T / self.steps
        # poisson- distributed jumps
        jumps = np.random.poisson(self.lam * dt, size=size)

        poi_rv = np.multiply(jumps,
                             np.random.normal(self.m, self.v, size=size)).cumsum(axis=0)
        geo = np.cumsum(((self.r - self.sigma ** 2 / 2 - self.lam * (self.m + self.v ** 2 * 0.5)) * dt +
                         self.sigma * np.sqrt(dt) *
                         np.random.normal(size=size)), axis=0)

        return np.exp(geo + poi_rv) * self.S, jumps

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
        return pd.DataFrame(self.data, columns=['0'])

    def plot_path_jumps(self):
        plt.figure(figsize=(12, 10))
        plt.plot(self.data, c='blue', label='time-series')
        plt.plot(self.jumps_x, self.jumps_y, "o", c='red', label='jumps')
        plt.grid(True)
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title('Jump Diffusion Process')
        plt.legend(loc='best')
        plt.show()

    def plot_variations(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.suptitle('Merkmale')
        fig.set_size_inches(12, 10)

        ax1.plot(self.returns, label='Returns')
        ax2.plot(self.rv, label='Realized variation')
        ax1.grid(True)
        ax2.grid(True)
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.show()

    # value wird später durch den F1-Score berechnet, nehme besten F1-Score von Cutoff, das ist dann value
    def calc_cutoff(self, value):
        cutoff_returns = np.zeros(len(self.data))
        cutoff_rv = np.zeros(len(self.data))

        for item in range(len(cutoff_returns)):
            if self.returns[item] > value:
                cutoff_returns[item] = 1

        for item in range(len(cutoff_rv)):
            if self.rv[item] > value:
                cutoff_rv[item] = 1

        return pd.DataFrame({
            'CutOff returns': cutoff_returns,
            'CutOff rv': cutoff_rv
        })

    # cut stimmt noch nicht der cut wird später der wert sein. mit dem man den höchsten F1-Score bekommt
    def plot_cutoff(self):

        # cut = best F1-Score
        cut = 0.01

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
        fig.suptitle('CutOff')

        ax1.plot(self.returns)
        ax1.hlines(y=cut, xmin=0, xmax=len(self.data), colors='red', label='CutOff')
        ax1.hlines(y=cut * -1, xmin=0, xmax=len(self.data), colors='red')

        ax2.plot(self.rv)
        ax2.hlines(y=cut, xmin=0, xmax=len(self.data), colors='red', label='CutOff')
        ax2.hlines(y=cut * -1, xmin=0, xmax=len(self.data), colors='red')
