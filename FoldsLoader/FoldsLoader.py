import os
from Config.general import *

def ProjectRoot():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return str(os.path.dirname(script_dir))

def FoldForWindow(fold, window):
    X = []
    y = []
    for file in fold:
        with open(os.path.join(ProjectRoot(), DataFolder, file), 'r') as file:
            X_unsplit = file.readlines()[window]
            X_parts = [float(x) for x in X_unsplit.strip().split(",")]
            X.append(X_parts)
            y.append("deliberate" in file.name)

    return X, y

def LoadFoldsPaths():
    files_dict = {}
    for n in range(1, 11):
        files_with_number_n = []
        for filename in os.listdir(os.path.join(ProjectRoot(), FoldsPath)):
            if filename.endswith(f"_{n}.txt"):
                with open(os.path.join(ProjectRoot(), FoldsPath, filename), 'r') as file:
                    files_with_number_n.extend([line.strip() + ".txt" for line in file.readlines()])
        files_dict[n] = files_with_number_n

    return files_dict

def LoadFold():
    paths = LoadFoldsPaths()
    Folds = []
    for w in range(WindowMax):
        WindowFolds = []
        for n in range(1, 11):
            fold = []
            X, y = FoldForWindow(paths[n], 0)
            fold.append(X)
            fold.append(y)
            WindowFolds.append(fold)
        Folds.append(WindowFolds)
    return Folds

def LoadFoldRanged():
    paths = LoadFoldsPaths()
    Folds = []
    return Folds

def LoadFoldPickOne():
    paths = LoadFoldsPaths()
    Folds = []
    return Folds