import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


def build_stock(stockDf, N=5,contamin=0.02,tage_pred:int=90):
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
    # Prediction
    stockDf['Prediction']=  np.where(stockDf["Close"].shift(-tage_pred) > stockDf["Close"], 1, 0)

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

    list  = model.fit_predict(data)
    ret = [1 if (i == -1) else 0 for i in list]

    return ret






