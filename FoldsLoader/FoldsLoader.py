import os
from Config.general import *
import numpy as np

from Config.persistanData import DataFolder, FoldsPath


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

def FoldForPickOne(fold, position):
    X = []
    y = []
    for file in fold:
        with open(os.path.join(ProjectRoot(), DataFolder, file), 'r') as file:
            X_tmp = file.readlines()
            X_total_lines = len(X_tmp)
            read_position = (X_total_lines - 1) * position // (WindowSections - 1)
            X_unsplit = X_tmp[read_position]
            X_parts = [float(x) for x in X_unsplit.strip().split(",")]
            X.append(X_parts)
            y.append("deliberate" in file.name)

    return X, y

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
            X, y = FoldForWindow(paths[n], w) #Był chyba błąd 0 zamiast zmiennej okna
            fold.append(X)
            fold.append(y)
            WindowFolds.append(fold)
        Folds.append(WindowFolds)
    return Folds

def LoadFoldRanged():
    paths = LoadFoldsPaths()
    Folds = []
    for w in range(WindowSections):
        WindowFolds = []
        for n in range(1, 11):
            fold = []
            X, y = FoldForPickRange(paths[n], w)
            fold.append(X)
            fold.append(y)
            WindowFolds.append(fold)
        Folds.append(WindowFolds)
    return Folds

def LoadFoldPickOne():
    paths = LoadFoldsPaths()
    Folds = []
    for w in range(WindowSections):
        WindowFolds = []
        for n in range(1, 11):
            fold = []
            X, y = FoldForPickOne(paths[n], w)
            fold.append(X)
            fold.append(y)
            WindowFolds.append(fold)
        Folds.append(WindowFolds)
    return Folds