from sklearn.model_selection import LeaveOneOut
from enum import Enum
from Config import WindowMax
from Config import WindowSections
from FoldsLoader import LoadFold
from FoldsLoader import LoadFoldPickOne
from FoldsLoader import LoadFoldRanged
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


class AppVariant(Enum):
    WindowNumber = 1
    WindowProportionalPickOne = 2
    WindowProportionalRange = 3


mode = AppVariant.WindowProportionalPickOne
windowsToProcess = 0

if mode == AppVariant.WindowNumber:
    windowsToProcess = WindowMax
    data = LoadFold()
elif mode == AppVariant.WindowProportionalPickOne:
    windowsToProcess = WindowSections
    data = LoadFoldPickOne()
else:  # mode == AppVariant.WindowProportionalRange
    windowsToProcess = WindowSections
    data = LoadFoldRanged()


def CalculateAccuracy(foldAccuracy):
    return sum(foldAccuracy) / len(foldAccuracy)


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
        foldAccuracy.append(metrics.accuracy_score(Y_test, Y_pred))

    accuracy.append(CalculateAccuracy(foldAccuracy))

print(accuracy)
plt.figure(figsize=(10, 6))
plt.plot(range(WindowSections), accuracy, marker='o', linestyle='-')
plt.title('Accuracy Across Different Windows')
plt.xlabel('Window Index')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
