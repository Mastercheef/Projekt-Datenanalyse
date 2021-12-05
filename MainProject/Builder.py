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


def buildMertonDF(jump_rate:float=None, l:int=None, step:int=None):
    # parameter mertion
    steps = 10000 if step == None else step
    lam = jump_rate * steps if l == None else l
    # generate merton data
    mertonData, jumps, contamin = merton_jump_paths(v=0.05, lam=lam, steps=steps)
    mertonDf = pd.DataFrame(mertonData, columns=['Merton Jump'])

    # add jumps
    jumps_x = list(np.ndarray.nonzero(jumps))[0]
    jumpsDf = pd.DataFrame(mertonDf.iloc[jumps_x])
    mertonDf['Jumps plot'] = 0
    mertonDf.loc[jumps_x, 'Jumps plot'] = jumpsDf['Merton Jump']

    jumps = [-1 if i > 0 else 1 for i in mertonDf['Jumps plot'].tolist()]
    mertonDf['Jumps'] = jumps

    # add features
    # log return
    mertonDf['Return log'] = np.log(mertonDf['Merton Jump']) - np.log(mertonDf['Merton Jump'].shift(1))

    N = 1  # Summ limit at RV and BPV

    # Realized variance
    mertonDf['RV'] = mertonDf['Return log'] ** 2
    mertonDf['RV'] = mertonDf['RV'].rolling(window=N).sum()
    # Bipower variance
    mertonDf['BPV'] = (np.log(mertonDf['Merton Jump']).shift(-1) - np.log(mertonDf['Merton Jump'])).abs() * mertonDf[
        'Return log'].abs()
    mertonDf['BPV'] = mertonDf['Return log'].abs() * (
                np.log(mertonDf['Merton Jump']) - np.log(mertonDf['Merton Jump'].shift(-1))).abs()
    mertonDf['BPV'] = mertonDf['BPV'].rolling(window=N).sum() * (np.pi / 2)
    mertonDf = mertonDf.dropna()
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
    # Achtung muss ungeding geprüft werden !!
    mertonDf['RSV'] = RV_pos

    # IF and features
    mertonDf['Anomaly Returns IF'] = isolationForest(mertonDf[['Return log']], contamin=contamin)
    mertonDf['Anomaly RSV IF'] = isolationForest(mertonDf[['RSV']], contamin=contamin)
    mertonDf['Anomaly Diff IF'] = isolationForest(mertonDf[['Diff']], contamin=contamin)
    mertonDf['Amomaly RSV Diff'] = isolationForest(mertonDf[['RSV', 'Diff']], contamin=contamin, max_features=2)
    mertonDf['Amomaly Returns RSV Diff'] = isolationForest(mertonDf[['Return log', 'RSV', 'Diff']], contamin=contamin,
                                                           max_features=3)

    return mertonDf


def subset(data=None):
    '''
    :param data:
    :return:
    '''
    subset_diff = data.loc[(data['Jumps'] == -1) | (data['Anomaly Diff IF'] == -1)]
    subset_rsv = data.loc[(data['Jumps'] == -1) | (data['Anomaly RSV IF'] == -1)]

    subset_diff = subset_diff[['Jumps', 'Anomaly Diff IF']]
    subset_rsv = subset_rsv[['Jumps', 'Anomaly RSV IF']]

    erg_diff = subset_diff.loc[(subset_diff['Jumps'] == -1) & (subset_diff['Jumps'] == subset_diff['Anomaly Diff IF'])]
    erg_rsv = subset_rsv.loc[(subset_rsv['Jumps'] == -1) & (subset_rsv['Jumps'] == subset_rsv['Anomaly RSV IF'])]

    erg_diff = erg_diff.count().loc['Jumps']
    erg_rsv = erg_rsv.count().loc['Jumps']

    outlier = len(subset_diff[subset_diff['Jumps'] == -1])

    percent_diff = round(erg_diff / outlier, 2) * 100
    percent_rsv = round(erg_rsv / outlier, 2) * 100
    contamin = len(subset_diff.loc[subset_diff['Anomaly Diff IF'] == -1])

    print('Diff: {} von {} Anomalien wurden erkannt -> {} % IF contamin: {}'.format(erg_diff, outlier, percent_diff,
                                                                                    contamin))
    print('RSV : {} von {} Anomalien wurden erkannt -> {} % IF contamin: {}'.format(erg_rsv, outlier, percent_rsv,
                                                                                    contamin))
    return percent_diff, subset_diff


def cutOff(data=None, label:str=None):
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
        cutoff_jump = [-1 if i > step or i < (step*(-1)) else 1 for i in data_list]
        df_tmp['Cutoff Jump'] = cutoff_jump
        f1 = f1_score(data['Jumps'], df_tmp['Cutoff Jump'], pos_label=-1)
        if f1 > bestF1:
            bestF1 = f1
            bestCutOff = step
            cutoff_list = cutoff_jump

    cutOff_ret['Cutoff Jump'] = cutoff_list
    return bestF1, bestCutOff, cutOff_ret


def isolationForest(data: [str], contamin: float, max_features: int = 1):
    """
    :param data:
    :param contamin:
    :return: a DF of anomaly valus where 1 = normal value and -1 = outlier
    """
    model = IsolationForest(n_estimators=100,
                            max_samples='auto',
                            contamination=contamin,
                            max_features=max_features)

    anomalyIF = model.fit_predict(data)

    return anomalyIF


def f1_score_comp(data=None, label: str = None):
    '''Computes the f1 score of an given DataFrame with positv_label = -1
    :param data:  dataset
    :param label: feature name
    :return: f1 score
    '''
    return f1_score(data['Jumps'], data[label], pos_label=-1)


def simulation(jump_rate, print=False):
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

    if print:
        print('F1 return log ', round(f1_ret_log, 3))
        # print('F1 rv ', round(f1_rvs,3))
        print('F1 diff ', round(f1_diff, 3))
        print('Cutoff return log ', round(cut_f1_ret_log, 3))
        # print('Cutoff RV ', round(cut_f1_rsv,3))
        print('Cutoff Diff ', round(cut_f1_diff, 3))

    return f1_ret_log, f1_diff, cut_f1_ret_log, cut_f1_diff, f1_rsv, cut_f1_rsv, rsv_diff, ret_rsv_diff


def plot_cut(data, label):
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


def plotter(df):
    plot_jumps = df[df['Jumps plot'] > 0]
    # plot Time series with jumps
    plt.figure(figsize=(18, 10))
    sns.lineplot(data=df['Merton Jump'], legend='auto', label='Time-series')
    sns.scatterplot(data=plot_jumps['Merton Jump'], label='Jumps', color='red', alpha=1, s=80)

    # Diff IF points
    diff = df.loc[(df['Anomaly Diff IF'] == -1)]
    diff = diff['Merton Jump']
    sns.scatterplot(data=diff, label='IF Diff', color='green', alpha=1, marker="+", s=120)
    # RSV IF points
    rsv = df.loc[(df['Anomaly RSV IF'] == -1)]
    rsv = rsv['Merton Jump']
    sns.scatterplot(data=rsv, label='IF RSV', color='orange', alpha=1, marker="v", s=120)

    # CutOff Return log
    cut_f1_ret_log, c1,cut = cutOff(df, 'Return log')
    cut = cut.loc[(cut['Cutoff Jump']) == -1]
    cut = cut['Merton Jump']
    sns.scatterplot(data=cut, label='CutOff Return', color='yellow', alpha=1, marker="v", s=120)

    # Returns log
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df['Return log'], legend='auto', label='Returns (log)')

    # plot features
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    fig.suptitle('Merkmale')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    sns.lineplot(ax=axes[0], data=df['BPV'], legend='auto', label='Bipower variation')
    sns.lineplot(ax=axes[1], data=df['RV'], legend='auto', label='Realized variation')
    sns.lineplot(ax=axes[2], data=df['Diff'], legend='auto', label='Difference')
    sns.lineplot(ax=axes[3], data=df['SJ'], legend='auto', label='Signed jumps')

    plt.show()


if __name__ == "__main__":
    df = buildMertonDF()
