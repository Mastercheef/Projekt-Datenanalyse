from sklearn.metrics import f1_score
import Builder
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

f1_scores_DiffIF = []
f1_scores_ReturnsIF = []
runs = 100

for i in range(runs):
    data = Builder.buildMertonDF()
    f1_scores_DiffIF.append(f1_score(data['Jumps'], data['Anomaly Diff IF'], pos_label=-1))
    f1_scores_ReturnsIF.append(f1_score(data['Jumps'], data['Anomaly Returns IF'], pos_label=-1))
    if (i % 10) == 0:
        print(i, "% erreicht")

end = time.time()
print("F1 Scores Mean Diff:", np.mean(f1_scores_DiffIF))
print("F1 Scores Mean Returns: ", np.mean(f1_scores_ReturnsIF))
print("Durchläufe:", runs)
print('Laufzeit: {:5.3f}s'.format(end - start))

plt.plot(f1_scores_ReturnsIF, label='Returns')
plt.plot(f1_scores_DiffIF, label='Diff')
plt.legend()
