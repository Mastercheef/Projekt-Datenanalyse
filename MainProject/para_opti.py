from Builder import simulation_test, plotter, plot_cut
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import Builder
import time
import sys

def optimierer(data: [str], contamin: float, i=0):
    model = IsolationForest(n_estimators=100,
                            max_samples=i,
                            contamination=contamin,
                            max_features=1,
                            bootstrap=False,
                            n_jobs=1,
                            random_state=0)
    list  = model.fit_predict(data)
    ret = [1 if (i == -1) else 0 for i in list]

    return ret

def para_opti():
    jump_steps = [0.0002, 0.001, 0.002, 0.005,0.01, 0.02]
    opti = []


    label = 'Return log'

    hallo = pd.DataFrame()
    start = time.time()
    for conta in jump_steps:
        print('step: ', conta)
        for _ in range(0,5):
            data = Builder.buildMertonDF(jump_rate=conta)
            max = 0
            wert_i = 0
            for i in range(2,120):
                hallo[label] = optimierer(data = data[[label]],contamin=conta,i=i)
                f1 = f1_score(data['Jumps'], hallo[label])
                if f1> max:
                    max = f1
                    wert_i = i
            opti.append(wert_i)
    end = time.time()
    sek = end - start
    print('running time: {} min'.format(round(sek/60,2)))
    print('Parameter für Diff: 10 pro rate: ',opti)



def subset(data=None):
    ''' Prints how many anomalies were detected with Diff and RSV.
    :param data: dataset [DataFrame]
    :return:
    '''
    subset_diff = data.loc[(data['Jumps'] == 1) | (data['Anomaly Diff IF'] == 1)]
    subset_cut_diff = data.loc[(data['Jumps'] == 1) | (data['CutOff Diff'] == 1)]
    subset_rsv = data.loc[(data['Jumps'] == 1) | (data['Anomaly RSV IF'] == 1)]
    subset_diff = subset_diff[['Jumps', 'Anomaly Diff IF']]
    subset_cut_diff = subset_cut_diff[['Jumps', 'CutOff Diff']]
    subset_rsv = subset_rsv[['Jumps', 'Anomaly RSV IF']]

    erg_diff = subset_diff.loc[(subset_diff['Jumps'] == 1) & (subset_diff['Jumps'] == subset_diff['Anomaly Diff IF'])]
    erg_cut_diff = subset_cut_diff.loc[(subset_cut_diff['Jumps'] == 1) & (subset_cut_diff['Jumps'] == subset_cut_diff['CutOff Diff'])]
    erg_rsv = subset_rsv.loc[(subset_rsv['Jumps'] == 1) & (subset_rsv['Jumps'] == subset_rsv['Anomaly RSV IF'])]

    erg_diff = erg_diff.count().loc['Jumps']
    erg_cut_diff = erg_cut_diff.count().loc['Jumps']
    erg_rsv = erg_rsv.count().loc['Jumps']

    outlier = len(subset_diff[subset_diff['Jumps'] == 1])

    percent_diff = round(erg_diff / outlier, 2) * 100
    percent_cut_diff = round(erg_cut_diff / outlier, 2) * 100
    percent_rsv = round(erg_rsv / outlier, 2) * 100

    contamin = len(subset_diff.loc[subset_diff['Anomaly Diff IF'] == 1])
    contamin_cut_diff = len(subset_cut_diff.loc[subset_cut_diff['CutOff Diff'] ==1])
    outlier_count = data[['Jumps','Anomaly Returns IF','CutOff Return', 'CutOff RSV', 'CutOff Diff']]
    print(outlier_count[outlier_count>0].count())
    print('-----------------------------')

    print('CutOF with Diff: {} of {} anomalies were recognized ->  {} %'.format(erg_cut_diff, outlier, round(percent_cut_diff,2)))
    print('Diff: {} of {} anomalies were recognized -> {} %'.format(erg_diff, outlier, round(percent_diff,2)))
    print('RSV : {} of {} anomalies were recognized -> {} %'.format(erg_rsv, outlier, round(percent_rsv,2)))
    print('-----------------------------')