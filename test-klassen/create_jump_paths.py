import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

sns.set_style('whitegrid')


class test_data():
    def __init__(self,lam=8):
        self.S   = 1.0  # current stock price
        self.T   = 1    # time to maturity
        self.r   = 0.02 # risk free rate (0.02)
        self.m   = 0    # meean of jump size
        self.v   = 0.1  # standard deviation of jump (0.3)
        self.lam = lam  # intensity of jump i.e. number of jumps per annum
        self.steps  = 1000 # time steps
        self.Npaths = 1     # number of paths to simulate
        self.sigma  = 0.4   # annaul standard deviation , for weiner process(0.2)

        self.data, self.jumps   = self.merton_jump_paths()
        self.data_df = self.comp_data_df()

        self.returns = self.calc_returns()
        self.rv = self.calc_rv()
        self.bpv = self.calc_bpv()
        self.diff = self.calc_diff()


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
        returns['pct'] = self.data_df['0'].pct_change()
        returns['log_ret'] = np.log(self.data_df['0']) - np.log(self.data_df['0'].shift(1))
        returns = returns.dropna()
        return returns

    def calc_rv(self, N=2):
        ''' Das N muss angegeben werden
        :param N:
        :return:
        '''
        rv = pd.DataFrame()
        rv['RV'] = self.returns['log_ret'] ** 2
        # rv['RV'] = rv['RV'].expanding(min_periods=1).sum()
        rv['RV'] = rv['RV'].rolling(window=N).sum()
        rv = rv.dropna()
        return rv

    def calc_bpv(self, N=2):
        bpv = pd.DataFrame()
        bpv['BPV'] = self.returns['log_ret'].abs() * (np.log(self.data_df['0']) - np.log(self.data_df['0'].shift(-1))).abs()
        bpv['BPV'] = bpv['BPV'].rolling(window=N).sum() * (np.pi/2)
        bpv = bpv.dropna()
        return bpv

    def calc_diff(self):
        return self.rv['RV'] - self.bpv['BPV']

    def calc_sj(self):
        '''
        unvollständig!!!!
        :return:
        '''
        return self.rv['RV'].abs() - self.rv['RV'].abs()


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
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
        fig.suptitle('Merkmale')
        fig.set_size_inches(12, 10)

        ax1.plot(self.returns['log_ret'], label='Returns (log)')
        ax2.plot(self.rv, label='Realized variation')
        ax3.plot(self.bpv, label='Realized Bipower variation')
        ax4.plot(self.diff, label='Difference')

        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax3.legend(loc='best')
        ax4.legend(loc='best')
        plt.show()

    def calc_cutoff(self, value):
        ''' Value wird später durch den F1-Score berechnet, nehme besten F1-Score von Cutoff, das ist dann value

        :param value:
        :return:
        '''
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

    def plot_cutoff(self):
        ''' Cut stimmt noch nicht der cut wird später der wert sein. mit dem man den höchsten F1-Score bekommt

        :return:
        '''

        # cut = best F1-Score
        cut = 0.01

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
        fig.suptitle('CutOff')

        ax1.plot(self.returns['log_ret'])
        ax1.hlines(y=cut, xmin=0, xmax=len(self.data), colors='red', label='CutOff')
        ax1.hlines(y=cut * -1, xmin=0, xmax=len(self.data), colors='red')

        ax2.plot(self.rv)
        ax2.hlines(y=cut, xmin=0, xmax=len(self.data), colors='red', label='CutOff')
        ax2.hlines(y=cut * -1, xmin=0, xmax=len(self.data), colors='red')
