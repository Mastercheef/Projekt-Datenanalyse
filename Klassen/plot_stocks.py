import cufflinks as cf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  stocks import Stocks

cf.go_offline()
sns.set_style('whitegrid')


class PlotStocks(Stocks):
    def __init__(self,stocks):
        self.stocks = stocks

    def plot_stocks_plt(self):
        for name in self.stocks.names:
            self.stocks.df_stocks[name]['Close'].plot(figsize=(16,10),label=name)
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_stocks_df(self):
        self.stocks.df_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()

    def clustermap(self):
        sns.clustermap(self.stocks.df_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)

    def heatmap(self):
        sns.heatmap(self.stocks.df_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
