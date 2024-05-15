from sklearn.model_selection import LeaveOneOut

from Config import WindowMax
from FoldsLoader import LoadFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

data = LoadFold()

def CalculateAccuracy(foldAccuracy):
    return sum(foldAccuracy) / len(foldAccuracy)

accuracy = []
for windowIndex in range(WindowMax):
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
plt.plot(range(WindowMax), accuracy, marker='o', linestyle='-')
plt.title('Accuracy Across Different Windows')
plt.xlabel('Window Index')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()