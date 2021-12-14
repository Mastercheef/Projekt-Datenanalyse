import numpy as np
import pandas as pd
import pandas_datareader as web
import yfinance as yf
from datetime import datetime
import cufflinks as cf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Stocks:
    def __init__(self, wkns: [] = None, names: [] = None, start=None, stop=None):
        self.wkns = wkns
        self.names = names
        self.start = start
        self.stop = stop
        self.df_stocks = self.read_and_merge()
        # Muss geprüft werden!!!
        self.stocks =self.df_stocks


    def read_and_merge(self):
        df_list = []
        for wkn in self.wkns:
            stock = yf.Ticker(wkn)
            df = stock.history(start=self.start, end=self.stop)
            df_list.append(df)
        stocks = pd.concat(df_list, axis=1, keys=self.names)
        stocks.columns.names = ['Stock Ticker', 'Stock Info']
        return stocks

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
