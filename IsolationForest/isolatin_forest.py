import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class i_f():
    def __init__(self,data):
        self.data = self.create_model(data)

    def create_model(self, data):
        model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1), max_features=1.0)
        model.fit(data[['0']])

        print(model.get_params())

        data['scores'] = model.decision_function(data[['0']])
        data['anomaly_score'] = model.predict(data[['0']])
        data[data['anomaly_score']==-1].head()

        return data


    def plot_if_jumps(self):
        plt.figure(figsize=(12,10))
        # data
        plt.plot(self.data['0'], c= 'blue',label='time-series')
        # anomalies
        plt.plot(self.data[self.data.anomaly_score<1]['0'],"o",c='red',label='jumps detectet by IF')

        plt.grid(True)
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title('Jump Diffusion Process')
        plt.legend(loc='best')
        plt.show()






