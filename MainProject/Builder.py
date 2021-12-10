import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from MertonJump import merton_jump_paths
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
sns.set()
sns.set_style('darkgrid')  # whitegrid


def buildMertonDF(jump_rate:float=None, l:int=None, step:int=None,S=1.0, T=1, r=0.01, m=0, v=0.055, lam=8, Npaths=1, sigma=0.45,N=1):
    ''' Creates a large data set with all features for the isolatin forest and associated anomaly values as well as the signed jumps.
    :param jump_rate: lambda/step (i.e. contamination) [float]
    :param l: lambda, intensity of jump [int]
    :param step: time steps, per default 10 000 [int]
    :return: dataset with Merton-jump-data,signed jumps, features, anomalie scores [Dataframe]
    '''
    # parameter mertion-jump
    steps = 10000 if step == None else step
    lam = jump_rate * steps if l == None else l

    # generate merton data
    mertonData, jumps, contamin = merton_jump_paths(S=S, T=T, r=r, m=m, v=v, lam=lam, steps=steps, Npaths=1, sigma=sigma)
    mertonDf = pd.DataFrame(mertonData, columns=['Merton Jump'])

    # add jumps
    jumps_x = list(np.ndarray.nonzero(jumps))[0]
    jumpsDf = pd.DataFrame(mertonDf.iloc[jumps_x])
    mertonDf['Jumps plot'] = 0
    mertonDf.loc[jumps_x, 'Jumps plot'] = jumpsDf['Merton Jump']
    jumps = [1 if i > 0 else 0 for i in mertonDf['Jumps plot'].tolist()]
    mertonDf['Jumps'] = jumps

    # add features
    # log return
    mertonDf['Return log'] = np.log(mertonDf['Merton Jump']) - np.log(mertonDf['Merton Jump'].shift(1))
    mertonDf = mertonDf.fillna(0)
    N = N  # Summ limit at RV and BPV

    # Realized variance
    mertonDf['RV'] = mertonDf['Return log'] ** 2
    mertonDf['RV'] = mertonDf['RV'].rolling(window=N).sum()
    mertonDf = mertonDf.fillna(0)
    # Bipower variance
    mertonDf['BPV'] = (np.log(mertonDf['Merton Jump']).shift(-1) - np.log(mertonDf['Merton Jump'])).abs() * (np.log(mertonDf['Merton Jump']) - np.log(mertonDf['Merton Jump'].shift(1))).abs()

    mertonDf['BPV'] = mertonDf['BPV'].rolling(window=N).sum() * (np.pi / 2)
    mertonDf = mertonDf.fillna(0)
    # Difference RV - BPV
    mertonDf['Diff'] = mertonDf['RV'] - mertonDf['BPV']

    # RV+ and RV-
    RV_pos = mertonDf[['Return log', 'RV']]
    RV_pos.loc[RV_pos['Return log'] < 0.0, 'RV'] = 0.0
    RV_pos = RV_pos['RV']

    RV_neg = mertonDf[['Return log', 'RV']]
    RV_neg.loc[RV_neg['Return log'] > 0.0, 'RV'] = 0.0
    RV_neg = RV_neg['RV']
    # Signed Jumps SJ
    mertonDf['SJ'] = RV_pos - RV_neg

    # Realized semi-variation RSV
    # realized semi-variation is referred to as signed jumps
    mertonDf['RSV'] = mertonDf['SJ']

    # IF and features
    mertonDf['Anomaly merton'] = isolationForest(mertonDf[['Merton Jump']], contamin=contamin)
    mertonDf['Anomaly Returns IF'] = isolationForest(mertonDf[['Return log']], contamin=contamin)
    mertonDf['Anomaly RSV IF'] = isolationForest(mertonDf[['RSV']], contamin=contamin)
    mertonDf['Anomaly Diff IF'] = isolationForest(mertonDf[['Diff']], contamin=contamin)
    mertonDf['Amomaly RSV Diff'] = isolationForest(mertonDf[['RSV', 'Diff']], contamin=contamin, max_features=2)
    mertonDf['Amomaly Returns RSV Diff'] = isolationForest(mertonDf[['Return log', 'RSV', 'Diff']], contamin=contamin,max_features=3)

    return mertonDf


def detected_anomalies(data=None):
    ''' Prints how many anomalies were detected with Diff and RSV.
    :param data: dataset [DataFrame]
    :return:
    '''
    for label in data.columns[9:].tolist():
        subset =  data.loc[(data['Jumps'] == 1) | (data[label] == 1)]
        erg_sub = subset.loc[(subset['Jumps'] == 1) & (subset['Jumps'] == subset[label])]
        erg_sub = erg_sub.count().loc['Jumps']

        outlier = len(subset[subset['Jumps'] == 1])
        loc_outlier = len(subset[subset[label]==1])
        pct = round(erg_sub / outlier, 2) * 100
        print(label+': {} of {} anomalies ({} %) in total: {}'.format(erg_sub, outlier, round(pct,2),loc_outlier))
        print('-----------------------------')


def cutOff(data=None, label:str=None):
    ''' This function calculates the cutoff of a given feature.
    :param data:  dataset  [DataFrame]
    :param label: feature  [string]
    :return: best F1-Score [int], CutOff-value [float], all data with a list of marked jumps [DataFrame]
    '''
    start = max(abs(data[label]))
    n = 100
    steps = np.linspace(start=start, stop=0, num=n)
    bestF1 = 0
    bestCutOff = 0
    cutoff_list = None
    df_tmp = pd.DataFrame()
    cutOff_df = data['Merton Jump']
    cutOff_ret = data
    data_list = data[label].values

    for step in steps:
        cutoff_jump = [1 if i > step or i < (step*(-1)) else 0 for i in data_list]
        df_tmp['Cutoff Jump'] = cutoff_jump
        f1 = f1_score(data['Jumps'], df_tmp['Cutoff Jump'])
        if f1 > bestF1:
            bestF1 = f1
            bestCutOff = step
            cutoff_list = cutoff_jump

    cutOff_ret['Cutoff Jump'] = cutoff_list
    return bestF1, bestCutOff, cutOff_ret


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


def f1_score_comp(data=None, label: str = None):
    '''Computes the f1 score of an given DataFrame with positv_label = -1
    :param data:  dataset [DataFrame]
    :param label: feature name [string]
    :return: f1 score [float]
    '''
    return f1_score(data['Jumps'], data[label])


def simulation_test(S=1.0, T=1, r=0.02, m=0, v=0.021, l=8, step=1000, Npaths=1, sigma=0.35,N=1,print_f1=False):
    data = buildMertonDF(S=S, T=T, r=r, m=m, v=v, l=l, step=step, Npaths=Npaths, sigma=sigma,N=N)
    # IF scores
    f1_ret_log = f1_score_comp(data, 'Anomaly Returns IF')
    f1_rsv = f1_score_comp(data, 'Anomaly RSV IF')
    f1_diff = f1_score_comp(data, 'Anomaly Diff IF')
    # Cutoff scores
    cut_f1_ret_log, c1,df1 = cutOff(data, 'Return log')
    cut_f1_rsv, c2, df2 = cutOff(data, 'RSV')
    cut_f1_diff, c3, df3 = cutOff(data, 'Diff')
    # multiple features
    rsv_diff = f1_score_comp(data, 'Amomaly RSV Diff')
    ret_rsv_diff = f1_score_comp(data, 'Amomaly Returns RSV Diff')

    data['CutOff Return'] = df1['Cutoff Jump']
    data['CutOff RSV'] = df2['Cutoff Jump']
    data['CutOff Diff'] = df3['Cutoff Jump']
    detected_anomalies(data)

    if print_f1:
        print('IF Return: ', round(f1_ret_log, 3))
        print('Cutoff Return: ', round(cut_f1_ret_log, 3))
        print('---------------------')
        print('IF Diff: ', round(f1_diff, 3))
        print('Cutoff Diff: ', round(cut_f1_diff, 3))
        print('---------------------')
        print('IF RSV: ', round(f1_rsv,3))
        print('Cutoff RSV: ', round(cut_f1_rsv,3))
        print('---------------------')
        print('IF RSV diff: ', round(rsv_diff, 3))
        print('IF Return RSV diff: ', round(ret_rsv_diff, 3))
        print('---------------------')

    return data


def simulation(jump_rate:float=None):
    ''' Simulates an analysis run with a random jump diffusion process with a given jump rate.
    :param jump_rate: contamination [float]
    :return: dataset [DataFrame], and feature scores [float]
    '''
    data = buildMertonDF(jump_rate=jump_rate)
    # IF scores
    f1_ret_log = f1_score_comp(data, 'Anomaly Returns IF')
    f1_rsv = f1_score_comp(data, 'Anomaly RSV IF')
    f1_diff = f1_score_comp(data, 'Anomaly Diff IF')
    # Cutoff scores
    cut_f1_ret_log, c1,df1 = cutOff(data, 'Return log')
    cut_f1_rsv, c2, df2 = cutOff(data, 'RSV')
    cut_f1_diff, c3, df3 = cutOff(data, 'Diff')
    # multiple features
    rsv_diff = f1_score_comp(data, 'Amomaly RSV Diff')
    ret_rsv_diff = f1_score_comp(data, 'Amomaly Returns RSV Diff')

    return data, f1_ret_log, f1_diff, cut_f1_ret_log, cut_f1_diff, f1_rsv, cut_f1_rsv, rsv_diff, ret_rsv_diff


def plot_cut(data=None, label:str=None):
    ''' Plots the specified feature as a line plot with an upper and lower cutOff.
    :param data: dataset [DataFrame]
    :param label: feature label [string]
    '''
    f1, cut, cutOff_df = cutOff(data, label)

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=data[label], legend='auto', label=label)

    c = [cut for i in range(1000)]
    c_min = [cut * (-1) for i in range(1000)]
    cut_df = pd.DataFrame(c, columns=['Cut'])
    cut_min_df = pd.DataFrame(c_min, columns=['Cut'])

    sns.lineplot(data=cut_df['Cut'], color='red', label='CutOff')
    sns.lineplot(data=cut_min_df['Cut'], color='red')
    plt.show()


def plotter(df=None):
    ''' Graphic example output of a merton-jump-diffusion process with signed anomalies and detected anomalies, as well as the feature output and Cutoff.
    :param df: datset with features and signed jumps [DataFrame]
    '''
    plot_jumps = df[df['Jumps plot'] > 0]
    # plot Time series with jumps
    plt.figure(figsize=(14, 12))
    sns.lineplot(data=df['Merton Jump'], legend='auto', label='Time-series')
    sns.scatterplot(data=plot_jumps['Merton Jump'], label='Jumps', color='red', alpha=1, s=80)

    # IF Diff anomalies
    diff = df.loc[(df['Anomaly Diff IF'] == 1)]
    diff = diff['Merton Jump']
    sns.scatterplot(data=diff, label='IF Diff', color='green', alpha=.6, marker="v", s=110)
    # RSV IF points
    rsv = df.loc[(df['Anomaly RSV IF'] == 1)]
    rsv = rsv['Merton Jump']
    #sns.scatterplot(data=rsv, label='IF RSV', color='orange', alpha=1, marker="v", s=120)

    # CutOff Diff anomalies
    cut_f1_ret_log, c1,cut = cutOff(df, 'Diff')
    cut = cut.loc[(cut['Cutoff Jump']) == 1]
    cut = cut['Merton Jump']
    sns.scatterplot(data=cut, label='Diff', color='yellow', alpha=0.6, marker="v", s=90)

    # Returns log
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df['Return log'], legend='auto', label='Returns (log)')

    # plot features
    fig, axes = plt.subplots(4, 1, figsize=(9, 12))
    fig.suptitle('Merkmale')
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    sns.lineplot(ax=axes[0], data=df['BPV'], legend='auto', label='Bipower variation')
    sns.lineplot(ax=axes[1], data=df['RV'], legend='auto', label='Realized variation')
    sns.lineplot(ax=axes[2], data=df['Diff'], legend='auto', label='Difference')
    sns.lineplot(ax=axes[3], data=df['SJ'], legend='auto', label='Signed jumps')

    plt.show()



