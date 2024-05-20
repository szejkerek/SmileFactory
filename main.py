from sklearn.model_selection import LeaveOneOut
from Config import AppVariant, mode, chosenClassifiers
from Config import WindowSections
from Config.persistanData import ClassifierVariant
from FoldsLoader import LoadFoldPickOne, LoadFoldRanged
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


current_time = datetime.now()
folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
data_dir = "Data"
folder_path = os.path.join(data_dir, folder_name)
os.makedirs(folder_path)


stdDevs = []
windowsToProcess = WindowSections
if mode == AppVariant.Window_Pick_One:
    data = LoadFoldPickOne()
else:  # mode == AppVariant.Window_Range
    data = LoadFoldRanged()

pickOneOrRangeMode = mode == AppVariant.Window_Pick_One or mode == AppVariant.Window_Range

classifiers = [
    (ClassifierVariant.RandomForest, 'Random Forest', RandomForestClassifier(n_estimators=100)),
    (ClassifierVariant.DecisionTree, 'Decision Tree', DecisionTreeClassifier(max_depth=5)),
    (ClassifierVariant.SVM, 'SVM', SVC(kernel='linear', probability=True)),
    (ClassifierVariant.KNN, 'KNN', KNeighborsClassifier(n_neighbors=5)),
    (ClassifierVariant.NeuralNet, 'Neural Net', MLPClassifier(alpha=1, max_iter=1000)),
    (ClassifierVariant.NaiveBayes, 'Naive Bayes', GaussianNB())
]


def GetClassifier(classifierVariant):
    for clf_type, clf_name, classifier in classifiers:
        if clf_type == classifierVariant:
            return clf_name, classifier
    return None, None


def CalculateAccuracy(foldMetrics):
    temp = {}
    for metric in foldMetrics:
        for key, value in metric.items():
            if key not in temp:
                temp[key] = []
            temp[key].append(value)

    result = {}
    for key, value in temp.items():
        result[key] = (np.mean(value), np.std(value))

    return result


def PickedMetric(Y_test, Y_pred):
    results = {}
    results['Accuracy'] = metrics.accuracy_score(Y_test, Y_pred)
    results['Precision'] = metrics.precision_score(Y_test, Y_pred)
    results['Recall'] = metrics.recall_score(Y_test, Y_pred)
    results['F1'] = metrics.f1_score(Y_test, Y_pred)
    results['ROC'] = metrics.roc_auc_score(Y_test, Y_pred)
    results['LogLoss'] = metrics.log_loss(Y_test, Y_pred)
    return results


def CalculateMatrics(cl):
    clf_name, classifier = GetClassifier(cl)
    metricsResults = []
    for windowIndex in range(windowsToProcess):
        X = data[windowIndex]
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        foldAccuracy = []
        print(f'Window {"range " if pickOneOrRangeMode else ""}index: {windowIndex} for classifier: {clf_name}')
        for i, (train_index, test_index) in enumerate(loo.split(X)):
            X_train = []
            Y_train = []
            for trainID in train_index:
                X_temp, Y_temp = X[trainID]
                X_train += X_temp
                Y_train += Y_temp
            X_test, Y_test = X[test_index[0]]

            clf = classifier
            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            foldAccuracy.append(PickedMetric(Y_test, Y_pred))

        metricsResults.append(CalculateAccuracy(foldAccuracy))
    return metricsResults

def SetUpSubPlot(axs, windowsCount, metric, std, title):
    xAxisLabels = windowsCount
    if pickOneOrRangeMode:
        xAxisLabels = [f'{(i / 7):.2f}-{((i + 1) / 7):.2f}' for i in
                       windowsCount]

    axs.plot(windowsCount, metric, marker='o', linestyle='-', color='b', label=f'{title}')
    axs.errorbar(windowsCount, metric, yerr=std, fmt='o', color='red', ecolor='orange', elinewidth=2, capsize=5,
                 label='Standard Deviation')

    axs.set_xticks(windowsCount)
    axs.set_xticklabels(xAxisLabels, rotation=20, ha='right')

    axs.set_title(title, fontsize= 16)

    axs.grid(True, linestyle='--', alpha=0.7)

    for i, value in enumerate(metric):
        axs.annotate(f'{value:.4f}', (windowsCount[i], value), textcoords="offset points", xytext=(0, 10), ha='center')
    if(np.mean(metric) <= 1):
        axs.set_ylim(0.5, 1)
    axs.set_facecolor('#f9f9f9')

    return axs


def CreatPlotFor(metrics, cl):
    clf_name, classifier = GetClassifier(cl)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    folds = list(range(len(metrics)))
    accuracy = [metric["Accuracy"][0] for metric in metrics]
    accuracy_std = [metric["Accuracy"][1] for metric in metrics]
    precision = [metric["Precision"][0] for metric in metrics]
    precision_std = [metric["Precision"][1] for metric in metrics]
    recall = [metric["Recall"][0] for metric in metrics]
    recall_std = [metric["Recall"][1] for metric in metrics]
    f1_score = [metric["F1"][0] for metric in metrics]
    f1_score_std = [metric["F1"][1] for metric in metrics]
    roc_auc = [metric["ROC"][0] for metric in metrics]
    roc_auc_std = [metric["ROC"][1] for metric in metrics]
    log_loss = [metric["LogLoss"][0] for metric in metrics]
    log_loss_std = [metric["LogLoss"][1] for metric in metrics]

    axs[0][0] = SetUpSubPlot(axs[0][0], folds, accuracy, accuracy_std, "Accuracy")

    axs[0][1] = SetUpSubPlot(axs[0][1], folds, precision, precision_std, "Precision")

    axs[0][2] = SetUpSubPlot(axs[0][2], folds, recall, recall_std, "Recall")

    axs[1][0] = SetUpSubPlot(axs[1][0], folds, f1_score, f1_score_std, "F1 Score")

    axs[1][1] = SetUpSubPlot(axs[1][1], folds, roc_auc, roc_auc_std, "ROC")

    axs[1][2] = SetUpSubPlot(axs[1][2], folds, log_loss, log_loss_std, "Log Loss")

    fig.suptitle(clf_name, fontsize=40)
    fig.tight_layout()

    plt.subplots_adjust(left=None, bottom=0.06, right=None, top=0.9, wspace=None, hspace=0.2)
    plt.savefig(os.path.join(folder_path, f'{clf_name}.png'))




for cl in chosenClassifiers:
    CreatPlotFor(CalculateMatrics(cl), cl)