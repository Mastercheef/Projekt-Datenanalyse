import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from MertonJump import merton_jump_paths


def buildMertonDF(S=100, T=1, r=0.02, m=0, v=0.3, lam=5, steps=1000, Npaths=1, sigma=0.2, rollingWindowRV=2):
    """
    This function generates
    :param rollingWindowRV:
    :param S: current stock price
    :param T: time to maturity
    :param r: risk free rate
    :param m: mean of jump size
    :param v: standard deviation of jump
    :param lam: intensity of jump i.e. number of jumps per annum
    :param steps: time steps
    :param Npaths: number of paths to simulate
    :param sigma: annaul standard deviation , for weiner process
    :return: dataframe with the generated stockprices, and the features
    """

    # generate merton data
    mertonData, jumps = merton_jump_paths(S, T, r, m, v, lam, steps, Npaths, sigma)

    mertonDf = pd.DataFrame(mertonData, columns=['Merton Jump'])

    # mertonDf['Return'] = mertonDf['Merton Jump'].pct_change()
    mertonDf['Return log'] = np.log(mertonDf['Merton Jump']) - np.log(mertonDf['Merton Jump'].shift(1))

    mertonDf['Realized variance'] = mertonDf['Return log'].rolling(rollingWindowRV).var()
    mertonDf = mertonDf.fillna(0)
    # mertonDf['RSV'] = ....
    # mertonDf['Diff'] = ...

    # cutoff
    # bestF1Returns = bestF1Score(mertonDf['Return'], jumps)
    # mertonDf['Anomaly Returns CutOff'] = cutOff(data=mertonDf['Return'], value=bestF1Returns)

    # bestF1RV = bestF1Score(mertonDf['Realized variance'], jumps)
    # mertonDf['Anomaly RV CutOff'] = cutOff(data=mertonDf['Realized variance'], value=bestF1RV)

    # mertonDf['Anomaly RSV CutOff'] = bestCutOff(data=mertonDf['Return'], value=0.02)
    # mertonDf['Anomaly Diff CutOff'] = bestCutOff(data=mertonDf['Return'], value=0.02)

    mertonDf['Anomaly Returns IF'] = isolationForest(mertonDf['Return log'])
    mertonDf['Anomaly RV IF'] = isolationForest(mertonDf['Realized variance'])
    # mertonDf['Anomaly RSV IF'] = isolationForest(mertonDF['RSV']
    # mertonDf['Anomaly Diff IF'] = isolationForest(mertonDF['Diff']

    return mertonDf


def cutOff(data, value):
    """

    :param data:
    :param value:
    :return:
    """
    # calc best f1-score = value
    cutoff_returns = np.zeros(len(data))
    for item in range(len(data)):
        if abs(data[item]) > value:
            cutoff_returns[item] = -1
        else:
            cutoff_returns[item] = 1
    return cutoff_returns


def isolationForest(data):
    """

    :param data:
    :return:
    """
    model = IsolationForest(n_estimators=50,
                            max_samples='auto',
                            contamination=float(0.1),
                            max_features=1.0)
    anomalyIF = model.fit_predict(np.array(data).reshape(len(data), 1))
    return anomalyIF


def bestF1Score(data, jumps):
    """

    :param data: array with one of the data features
    :param jumps: array of the merton jumps, ytrue
    :return: best F1-Score for the cutoff method
    """
    """
    jumps = [0,0,0,1,0,1,0]
    data  = [0,1,0,1,0,1,0]
    starte als value für Cutoff bei dem betrag höchsten wert, laufe bis 0,
    irgendwo dazwischen ist der Cutoff der den besten CutOff wert gibt, 
    nehme den besten CutOff als finalen wert
    """
    start = max(abs(data))
    n = 1000
    steps = np.linspace(start=start, stop=0, num=n)
    bestF1 = 0
    bestCutOff = 0

    for step in steps:
        if f1_score(y_true=jumps, y_pred=cutOff(data, step), average=None) > bestF1:
            bestCutOff = step

    return bestCutOff


def plotter(data, plot):
    pass


buildMertonDF()