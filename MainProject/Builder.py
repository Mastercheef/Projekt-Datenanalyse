import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

from MertonJump import merton_jump_paths

'''

'''


def buildMertonDF(S=100, T=1, r=0.02, m=0, v=0.3, lam=5, steps=1000, Npaths=1, sigma=0.2):
    mertonData, jumps = merton_jump_paths(S, T, r, m, v, lam, steps, Npaths, sigma)

    mertonDf = pd.DataFrame(mertonData, columns=['Merton Jump'])
    # mertonDf['Merton Jump'] = mertonData
    mertonDf['Return'] = mertonDf['Merton Jump'].pct_change()

    mertonDf['Realized variance'] = mertonDf['Return'].rolling(2).var()
    mertonDf = mertonDf.fillna(0)
    # mertonDf['RSV'] = ....
    # mertonDf['Diff'] = ...

    mertonDf['Anomaly Returns CutOff'] = bestCutOff(data=mertonDf['Return'], value=0.01)
    mertonDf['Anomaly RV CutOff'] = bestCutOff(data=mertonDf['Realized variance'], value=0.02)
    # mertonDf['Anomaly RSV CutOff'] = bestCutOff(data=mertonDf['Return'], value=0.02)
    # mertonDf['Anomaly Diff CutOff'] = bestCutOff(data=mertonDf['Return'], value=0.02)

    mertonDf['Anomaly Returns IF'] = isolationForest(mertonDf['Return'])
    mertonDf['Anomaly RV IF'] = isolationForest(mertonDf['Realized variance'])
    # mertonDf['Anomaly RSV IF'] = isolationForest(mertonDF['RSV']
    # mertonDf['Anomaly Diff IF'] = isolationForest(mertonDF['Diff']

    return mertonDf


def bestCutOff(data, value):
    # calc best f1-score = value
    cutoff_returns = np.zeros(len(data))
    for item in range(len(data)):
        if abs(data[item]) > value:
            cutoff_returns[item] = -1
        else:
            cutoff_returns[item] = 1
    return int(cutoff_returns)


def isolationForest(data):
    model = IsolationForest(n_estimators=50,
                            max_samples='auto',
                            contamination=float(0.1),
                            max_features=1.0)
    anomalyIF = model.fit_predict(np.array(data).reshape(len(data), 1))
    return anomalyIF


def plotter(data, plot):
    pass
