import pandas as pd
import numpy as np
import time
from stocks import *
from plot_stocks import *
from Builder_stock import acc_score, build_stock
from Plotter_stock import *

def stock_simulaiton():
    branchen = ['EOAN.DE','RWE.DE','ALV.DE','MUV2.DE','DBK.DE','CBK.DE','NWT.F','AFX.DE','PFE.F','SIX2.DE','DTE.DE','SOBA.F','VODI.DE','TNE5.F','SIE.DE','IFX.DE',
                'SAP.DE','ZAL.DE','FNTN.DE','ABEC.DE','VOW3.DE','DAI.DE','ADS.DE','PSM.DE','BOSS.DE','CCC3.DE','PRG.F','LIN.DE','HEI.DE','TKA.DE','TUI1.DE','BRH.F','BAS.DE','BAYN.DE','WMT.F']

    stocks = Stocks(branchen, branchen, start="2015-01-01", stop="2021-10-01")
    stocks = stocks.df_stocks.dropna()

    conta_list = [0.002,0.005,0.008,0.010,0.015,0.02]
    # conta_list_two = [0.10,0.20,0.5]

    a_score,c_score,mean_a,mean_c = [],[],[],[]
    start = time.time()
    for conta in conta_list:
        print('Conta', conta)
        for label in branchen:
            df = pd.DataFrame(stocks[label]['Close'])
            df = build_stock(df,N=1,contamin=conta,tage_pred=90)
            anom_score, comp_score = acc_score(df)
            a_score.append(anom_score)
            c_score.append(comp_score)
        mean_a.append(np.mean(a_score))
        mean_c.append(np.mean(c_score))

    end = time.time()
    sek = end - start
    print('running time: {} min'.format(round(sek/60,2)))
    table = pd.DataFrame(data = [conta_list,mean_a,mean_c])
    table = table.transpose()
    table.columns = ['Contamination', 'Anomaly Score', 'Random Score']

    return table