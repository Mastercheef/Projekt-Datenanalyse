import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from MertonJump import merton_jump_paths
sns.set()
sns.set_style('darkgrid') # whitegrid

def buildMertonDF(S=1.0, T=1, r=0.02, m=0, v=0.1, lam=8, steps=1000, Npaths=1, sigma=0.4, N=2):
    """ This function generates
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
    :param N: sum index for RV and BPV
    :return: dataframe with the generated stockprices, and the features
    """

    # generate merton data
    mertonData, jumps, contamin = merton_jump_paths(S, T, r, m, v, lam, steps, Npaths, sigma)
    mertonDf = pd.DataFrame(mertonData, columns=['Merton Jump'])
    # add jumps
    jumps_x = list(np.ndarray.nonzero(jumps))[0]
    jumpsDf = pd.DataFrame(mertonDf.iloc[jumps_x])
    mertonDf['Jumps plot'] = 0
    mertonDf.loc[jumps_x,'Jumps plot'] = jumpsDf['Merton Jump']

    jumps = [-1 if i>0 else 1 for i in mertonDf['Jumps plot'].tolist()]
    mertonDf['Jumps'] = jumps
    # add features

    # log return
    mertonDf['Return log'] = np.log(mertonDf['Merton Jump']) - np.log(mertonDf['Merton Jump'].shift(1))
    # Realized variance
    mertonDf['RV'] = mertonDf['Return log']**2
    mertonDf['RV'] = mertonDf['RV'].rolling(window=N).sum()
    # Bipower variance
    mertonDf['BPV'] = mertonDf['Return log'].abs() * (np.log(mertonDf['Merton Jump']) - np.log(mertonDf['Merton Jump'].shift(-1))).abs()
    mertonDf['BPV'] = mertonDf['BPV'].rolling(window=N).sum() * (np.pi/2)
    mertonDf = mertonDf.dropna()
    # Difference RV - BPV
    mertonDf['Diff'] = mertonDf['RV'] - mertonDf['BPV']




    #mertonDf = mertonDf.fillna(0)


    # cutoff
    # bestF1Returns = bestF1Score(mertonDf['Return'], jumps)
    # mertonDf['Anomaly Returns CutOff'] = cutOff(data=mertonDf['Return'], value=bestF1Returns)

    # bestF1RV = bestF1Score(mertonDf['Realized variance'], jumps)
    # mertonDf['Anomaly RV CutOff'] = cutOff(data=mertonDf['Realized variance'], value=bestF1RV)

    # mertonDf['Anomaly RSV CutOff'] = bestCutOff(data=mertonDf['Return'], value=0.02)
    # mertonDf['Anomaly Diff CutOff'] = bestCutOff(data=mertonDf['Return'], value=0.02)

    mertonDf['Anomaly Returns IF'] = isolationForest(mertonDf['Return log'],contamin=contamin)
    mertonDf['Anomaly RV IF'] = isolationForest(mertonDf['RV'],contamin=contamin)
    #mertonDf['Anomaly RSV IF'] = isolationForest(mertonDf['RSV'],contamin=contamin)
    mertonDf['Anomaly Diff IF'] = isolationForest(mertonDf['Diff'],contamin=contamin)

    return mertonDf

def subset(data):
    subset = data.loc[(data['Jumps']==-1) | (data['Anomaly Diff IF']==-1)]
    subset=subset[['Jumps', 'Anomaly Diff IF']]

    erg = subset.loc[(subset['Jumps']==-1)&(subset['Jumps']==subset['Anomaly Diff IF'])]
    erg = erg.count().loc['Jumps']
    outlier = len(subset[subset['Jumps']==-1])

    percent = round(erg/outlier,2)*100
    contamin = len(subset.loc[subset['Anomaly Diff IF']==-1])
    print('{} von {} Anomalien wurden erkannt -> {} % IF contamin: {}'.format(erg, outlier,percent, contamin ))
    return percent,subset




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


def isolationForest(data,contamin):
    """

    :param data:
    :return:
    """
    model = IsolationForest(n_estimators=100,
                            max_samples='auto',
                            contamination=contamin,
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


def plotter(df):
    plot_jumps =df[df['Jumps plot']>0]
    # plot Time series with jumps
    plt.figure(figsize=(18,10))
    sns.lineplot(data=df['Merton Jump'],legend='auto',label='Time-series')
    sns.scatterplot(data=plot_jumps['Merton Jump'], label='Jumps',color='red',alpha=1,s=80)

    subset = df.loc[(df['Anomaly Diff IF']==-1)]
    subset = subset['Merton Jump']

    sns.scatterplot(data=subset, label='IF Diff',color='green',alpha=1, marker="+",s=120)



    # plot features
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Merkmale')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    sns.lineplot(ax=axes[0],data=df['BPV'],legend='auto',label='Bipower variation')
    sns.lineplot(ax=axes[1],data=df['RV'],legend='auto',label='Realized variation')
    sns.lineplot(ax=axes[2],data=df['Diff'],legend='auto',label='Difference')

    plt.show()



if __name__ == "__main__":

    df = buildMertonDF()




