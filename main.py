from Config import WindowMax
from FoldsLoader import LoadFold

data = LoadFold()
for windowIndex in range(WindowMax):
    for foldIndex in range(0, 10):
        X, y = data[windowIndex][foldIndex]



# import numpy as np
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score


# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 3: Perform cross-validation
# scores = cross_val_score(rf_classifier, X, y, cv=5)
#
# # Step 4: Print cross-validation scores
# print("Cross-Validation Scores:", scores)
# print("Mean Accuracy:", np.mean(scores))
#
# # foldsW1 = [fold1, fold1,fold1,fold1,fold1,fold1,fold1,fold1,fold1,]
# # folsd = [foldsW1 --- W13]
#
# # scores = cross_val_score(rf_classifier, X, y, cv=5)
# #
# # # Train the random forest classifier
# # rf_classifier.fit(X, y)
# #
# # # Visualize one of the trees in the random forest
# # plt.figure(figsize=(12, 8))
# # plot_tree(rf_classifier.estimators_[0], filled=True)
# # plt.show()
