from Builder import simulation_test,subset, plotter, plot_cut
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


jump_steps = [0.0002, 0.001, 0.002, 0.005,0.01, 0.02]
opti = []


hallo = pd.DataFrame()
start = time.time()
for conta in jump_steps:
    print('step: ', conta)
    for _ in range(0,5):
        data = Builder.buildMertonDF(jump_rate=conta)
        max = 0
        wert_i = 0
        for i in range(2,120):
            hallo['Diff'] = optimierer(data = data[['Diff']],contamin=conta,i=i)
            f1 = f1_score(data['Jumps'], hallo['Diff'])
            if f1> max:
                max = f1
                wert_i = i
        opti.append(wert_i)
end = time.time()
sek = end - start
print('running time: {} min'.format(round(sek/60,2)))
print('Parameter für Diff: 10 pro rate: ',opti)