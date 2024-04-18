import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
import os

import pandas as pd

fold1P = 'Resources\\list\\P_fold_all_1.txt'
fold1S = 'Resources\\list\\S_fold_all_1.txt'


def LoadFold(path, window ,isDeliberate):
    file_path = 'Resources\\UvA-NEMO\\features\\cross-AU window13'
    y = []
    X = []
    with open(path, 'r') as file:
        content = file.readlines()

    for line in content:
        file_name = line.strip() + ".txt"
        full_path = os.path.join(file_path, file_name)

        with open(full_path, 'r') as file:
            X_unsplit = file.readlines()[window]
            X_parts = X_unsplit.strip().split(",")
            X.append(X_parts)
            y.append(isDeliberate)


    return X,y



Xp,yp = LoadFold(fold1P, 1,False)
Xs,ys = LoadFold(fold1S, 1,True)

X = Xp + Xs
y = yp + ys



rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 3: Perform cross-validation
scores = cross_val_score(rf_classifier, X, y, cv=5)

# Step 4: Print cross-validation scores
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))

# foldsW1 = [fold1, fold1,fold1,fold1,fold1,fold1,fold1,fold1,fold1,]
# folsd = [foldsW1 --- W13]

# scores = cross_val_score(rf_classifier, X, y, cv=5)
#
# # Train the random forest classifier
# rf_classifier.fit(X, y)
#
# # Visualize one of the trees in the random forest
# plt.figure(figsize=(12, 8))
# plot_tree(rf_classifier.estimators_[0], filled=True)
# plt.show()
