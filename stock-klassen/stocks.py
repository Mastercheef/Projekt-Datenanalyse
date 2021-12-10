import numpy as np
import pandas as pd
import pandas_datareader as web
import yfinance as yf
from datetime import datetime

class Stocks:
    def __init__(self, wkns: [] = None, names: [] = None, start=None, stop=None):
        self.wkns = wkns
        self.names = names
        self.start = start
        self.stop = stop
        self.df_stocks = self.read_and_merge()
        # features
        self.df_returns = self.returns()
        self.df_log_returns  = self.log_returns()
        self.df_rv = self.rv()
        self.df_bpv = self.bpv()
        self.df_diff = self.diff()

    def read_and_merge(self):
        df_list = []
        for wkn in self.wkns:
            stock = yf.Ticker(wkn)
            df = stock.history(start=self.start, end=self.stop)
            df_list.append(df)
        stocks = pd.concat(df_list, axis=1, keys=self.names)
        stocks.columns.names = ['Stock Ticker', 'Stock Info']
        return stocks

    def returns(self):
        returns = pd.DataFrame()
        for name in self.names:
            returns[name + ' Return'] = self.df_stocks[name]['Close'].pct_change()
        returns = returns.fillna(0)
        return returns

    def log_returns(self):
        log_returns = pd.DataFrame()
        for name in self.names:
            log_returns[name + ' log Return'] = np.log(self.df_stocks[name]['Close']) - np.log(self.df_stocks[name]['Close'].shift(1))
        log_returns = log_returns.fillna(0)
        return log_returns

    def rv(self,N=1):
        rv = pd.DataFrame()
        for name in self.names:
            rv[name + ' RV'] = self.df_log_returns[name + ' log Return'] ** 2
            rv[name + ' RV'] = rv[name + ' RV'].rolling(window=N).sum()
            rv = rv.fillna(0)
        return rv


    def bpv(self,N=1):
        bpv = pd.DataFrame()
        for name in self.names:
            bpv[name +' BPV'] = (np.log(self.df_stocks[name]['Close']).shift(-1) - np.log(self.df_stocks[name]['Close'])).abs() * self.df_returns[name+' Return log'].abs()
            bpv[name +' BPV'] = bpv[name +' BPV'].rolling(window=N).sum() * (np.pi / 2)
        bpv = bpv.fillna(0)
        return bpv


    def diff(self):
        return  self.df_rv - self.df_bpv

