from Config import WindowMax
from FoldsLoader import LoadFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Preparing variables
X_train = []
Y_train = []
X_test = []
Y_test = []
test_index = 8


data = LoadFold()
for windowIndex in range(WindowMax):
    for foldIndex in range(0, 10):
        X, Y = data[windowIndex][foldIndex]
        if foldIndex != test_index:
            X_train += X
            Y_train += Y
        else:
            X_test = X
            Y_test = Y



# creating a RF classifier
clf = RandomForestClassifier(n_estimators=100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(Y_test, Y_pred))


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
