from sklearn.model_selection import LeaveOneOut
from enum import Enum
from Config import WindowMax, AppVariant, mode, FoldSumaryMode, accuracyVariant, metricVariant, MetricVariant
from Config import WindowSections
from FoldsLoader import LoadFold, LoadFoldPickOne, LoadFoldRanged
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


stdDevs = []

if mode == AppVariant.Window_Number:
    windowsToProcess = WindowMax
    data = LoadFold()
elif mode == AppVariant.Window_Pick_One:
    windowsToProcess = WindowSections
    data = LoadFoldPickOne()
else:  # mode == AppVariant.Window_Range
    windowsToProcess = WindowSections
    data = LoadFoldRanged()



def CalculateAccuracy(foldAccuracy, accuracyVariant):
    if accuracyVariant == FoldSumaryMode.Max:
        result = max(foldAccuracy)
    elif accuracyVariant == FoldSumaryMode.Min:
        result = min(foldAccuracy)
    else:  # accuracyVariant == AccuracyVariant.Average
        result = np.mean(foldAccuracy)
        stdDevs.append(np.std(foldAccuracy))
    return result

def PickedMetric(Y_test, Y_pred):
    result = 0
    if metricVariant == MetricVariant.Accuracy:
        result = metrics.accuracy_score(Y_test, Y_pred)
    elif metricVariant == MetricVariant.Precision:
        result = metrics.precision_score(Y_test, Y_pred)
    elif metricVariant == MetricVariant.Recall:
        result = metrics.recall_score(Y_test, Y_pred)
    elif metricVariant == MetricVariant.F1:
        result = metrics.f1_score(Y_test, Y_pred)
    elif metricVariant == MetricVariant.ROC:
        result = metrics.roc_auc_score(Y_test, Y_pred)
    elif metricVariant == MetricVariant.LogLoss:
        result = metrics.log_loss(Y_test, Y_pred)
    return result



accuracy = []
for windowIndex in range(windowsToProcess):
    X = data[windowIndex]
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    foldAccuracy = []
    print(f'Window id: {windowIndex}')
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train = []
        Y_train = []
        for trainID in train_index:
            X_temp, Y_temp = X[trainID]
            X_train += X_temp
            Y_train += Y_temp
        X_test, Y_test = X[test_index[0]]

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        foldAccuracy.append(PickedMetric(Y_test, Y_pred))

    accuracy.append(CalculateAccuracy(foldAccuracy,accuracyVariant))

print(accuracy)
plt.figure(figsize=(10, 6))
plt.plot(range(windowsToProcess), accuracy, marker='o', linestyle='-', label=f'{metricVariant.name} {accuracyVariant.name}', color='b')
if accuracyVariant == FoldSumaryMode.Average:
    plt.errorbar(range(len(accuracy)), accuracy, yerr=stdDevs, fmt='o', color='red', ecolor='orange', elinewidth=2, capsize=5, label='Standard Deviation')
plt.title(f'{metricVariant.name} {accuracyVariant.name} Across Different {mode.name}', fontsize=16, fontweight='bold')
plt.xticks(range(len(accuracy)), fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0.5, 1)
plt.xlabel('Window', fontsize=14)
plt.ylabel(metricVariant.name)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=9)

for i, value in enumerate(accuracy):
    plt.annotate(f'{value:.4f}', (i, value), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

plt.gca().set_facecolor('#f9f9f9')
plt.tight_layout()
plt.show()
