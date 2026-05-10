# SmileFactory

## Code Highlights

### Proportional Temporal Interval Averaging

```python
def FoldForPickRange(fold, position):
    X = []
    y = []
    for file in fold:
        with open(os.path.join(ProjectRoot(), DataFolder, file), 'r') as file:
            X_tmp = file.readlines()
            X_total_lines = len(X_tmp)

            range_start_position = X_total_lines * position // WindowSections
            range_end_position = X_total_lines * (position + 1) // WindowSections

            X_tmp_unsplit = X_tmp[range_start_position]
            X_tmp_parts = [float(x) for x in X_tmp_unsplit.strip().split(",")]
            X_sum_parts = np.array(X_tmp_parts)

            for i in range(range_start_position + 1, range_end_position):
                X_tmp_unsplit = X_tmp[i]
                X_tmp_parts = [float(x) for x in X_tmp_unsplit.strip().split(",")]
                X_sum_parts = X_sum_parts + np.array(X_tmp_parts)

            X_sum_parts = X_sum_parts / (range_end_position - range_start_position)
            X_parts = X_sum_parts.tolist()
            X.append(X_parts)
            y.append("deliberate" in file.name)

    return X, y
```

Each smile is a variable-length sequence of high-dimensional AU feature vectors. This function divides that sequence into `WindowSections` proportional temporal intervals using integer arithmetic (`total * position // sections`), then averages all feature vectors within the selected interval into a single representative vector. The use of floor division ensures exact, non-overlapping partition boundaries regardless of sequence length, avoiding floating-point rounding issues. This is the core of the "ranged window" experiment mode, and contrasts with point sampling in `FoldForPickOne`.

### Leave-One-Out Evaluation with Flattened Fold Data

```python
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
```

For each temporal window position, this function performs Leave-One-Out cross-validation over 10 predefined subject-partitioned folds. Each training iteration flattens all samples from the 9 included folds into a single flat feature matrix before fitting the classifier. The held-out fold is used as the test set. After all LOO iterations, `CalculateAccuracy` computes per-metric mean and standard deviation, enabling the per-window performance plots that are the primary output of the pipeline.
