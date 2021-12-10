import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
sns.set()
sns.set_style('darkgrid')  # whitegrid


def build_stock(stockDf, N=5,contamin=0.02):
    stockDf['Return'] = stockDf['Close'].pct_change()
    stockDf['Return log'] = np.log(stockDf['Close']) - np.log(stockDf['Close'].shift(1))
    stockDf = stockDf.fillna(0)

    # Realized variance
    stockDf['RV'] = stockDf['Return log'] ** 2
    stockDf['RV'] = stockDf['RV'].rolling(window=N).sum()
    stockDf = stockDf.fillna(0)
    # Bipower variance
    stockDf['BPV'] = (np.log(stockDf['Close']).shift(-1) - np.log(stockDf['Close'])).abs() * (np.log(stockDf['Close']) - np.log(stockDf['Close'].shift(1))).abs()
    stockDf['BPV'] = stockDf['BPV'].rolling(window=N).sum() * (np.pi / 2)
    stockDf = stockDf.fillna(0)
    # Difference RV - BPV
    stockDf['Diff'] = stockDf['RV'] - stockDf['BPV']

    # RV+ and RV-
    RV_pos = stockDf[['Return log', 'RV']]
    RV_pos.loc[RV_pos['Return log'] < 0.0, 'RV'] = 0.0
    RV_pos = RV_pos['RV']
    RV_neg = stockDf[['Return log', 'RV']]
    RV_neg.loc[RV_neg['Return log'] > 0.0, 'RV'] = 0.0
    RV_neg = RV_neg['RV']
    # Signed Jumps SJ
    stockDf['SJ'] = RV_pos - RV_neg

    # Realized semi-variation RSV
    stockDf['RSV'] = stockDf['SJ']

    # IF and features

    stockDf['Anomaly Close'] = isolationForest(stockDf[['Close']], contamin=contamin)
    stockDf['Anomaly pct Return'] = isolationForest(stockDf[['Return']], contamin=contamin)
    stockDf['Anomaly Returns IF'] = isolationForest(stockDf[['Return log']], contamin=contamin)
    stockDf['Anomaly RSV IF'] = isolationForest(stockDf[['RSV']], contamin=contamin)
    stockDf['Anomaly Diff IF'] = isolationForest(stockDf[['Diff']], contamin=contamin)
    stockDf['Amomaly RSV Diff'] = isolationForest(stockDf[['RSV', 'Diff']], contamin=contamin, max_features=2)
    stockDf['Amomaly Returns RSV Diff'] = isolationForest(stockDf[['Return log', 'RSV', 'Diff']], contamin=contamin,max_features=3)

    return stockDf


def isolationForest(data: [str], contamin: float, max_features: int = 1):
    """ Creates an isolation forest based on the transferred data
    :param data: dataset [DataFrame]
    :param contamin: the jump-rate of the dataset [float]
    :param max_features:
    :return: dataset of anomaly valus where 0 = inlier and 1 = outlier [DataFrame]
    """

    model = IsolationForest(n_estimators=100,
                            max_samples=0.25,
                            contamination=contamin,
                            max_features=max_features,
                            random_state=11)
    #bootstrap=False,
    #n_jobs=1,

    list  = model.fit_predict(data)
    ret = [1 if (i == -1) else 0 for i in list]

    return ret


def plotter(df=None,label:str=None):
    ''' Graphic example output of a merton-jump-diffusion process with signed anomalies and detected anomalies, as well as the feature output and Cutoff.
    :param df: datset with features and signed jumps [DataFrame]
    '''

    plt.figure(figsize=(12, 9))
    sns.lineplot(data=df['Close'], legend='auto', label=label)

    # IF Return anomalies points
    ret = df.loc[(df['Anomaly Returns IF'] == 1)]
    ret_return = ret['Return log']
    ret = ret['Close']
    sns.scatterplot(data=ret, label='IF Return', color='red', alpha=.6, s=110)
    # IF Diff anomalies points
    diff = df.loc[(df['Anomaly Diff IF'] == 1)]
    diff_diff = diff['Diff']
    diff = diff['Close']
    sns.scatterplot(data=diff, label='IF Diff', color='green', alpha=.6, marker="v", s=70)
    # RSV IF anomalie points
    rsv = df.loc[(df['Anomaly RSV IF'] == 1)]
    rsv_rsv = rsv['SJ']
    rsv = rsv['Close']
    sns.scatterplot(data=rsv, label='IF RSV', color='orange', alpha=1, marker="v", s=70)


    # Return log
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df['Return log'], legend='auto', label='Returns (log)')
    sns.scatterplot(data=ret_return, legend= 'auto',label='IF Return', color='red', s=110)

    rsv_diff = df.loc[(df['Amomaly RSV Diff'] == 1)]
    rsv_diff = rsv_diff['SJ']
    # plot features
    fig, axes = plt.subplots(4, 1, figsize=(9, 12))
    fig.suptitle('Merkmale')
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    sns.lineplot(ax=axes[0], data=df['Return log'], legend='auto', label='Returns (log)')
    sns.scatterplot(ax=axes[0],data=ret_return,color='red', legend= 'auto', label='Anomaly')
    sns.lineplot(ax=axes[1], data=df['Diff'], legend='auto', label='Difference')
    sns.scatterplot(ax=axes[1],data=diff_diff,color='red', legend= 'auto', label='Anomaly')
    sns.lineplot(ax=axes[2], data=df['SJ'], legend='auto', label='RSV')
    sns.scatterplot(ax=axes[2],data=rsv_rsv,color='red', legend= 'auto', label='Anomaly')
    sns.lineplot(ax=axes[3], data=df['SJ'], legend='auto', label='RSV and Diff')
    sns.scatterplot(ax=axes[3],data=rsv_diff,color='red', legend= 'auto', label='Anomaly')


    plt.show()



