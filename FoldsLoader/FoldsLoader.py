import os
from Config.general import *

def ProjectRoot():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return str(os.path.dirname(script_dir))

def LoadFoldsDictionary():
    files_dict = {}
    for n in range(1, 11):
        files_with_number_n = []
        for filename in os.listdir(os.path.join(ProjectRoot(), FoldsPath)):
            if filename.endswith(f"_{n}.txt"):
                files_with_number_n.append(os.path.join(FoldsPath, filename))
        files_dict[n] = files_with_number_n

    return files_dict

def LoadFold():
    foldsDictionary = LoadFoldsDictionary()




# file_path = 'Resources\\UvA-NEMO\\features\\cross-AU window13'
# y = []
# X = []
# with open(path, 'r') as file:
#     content = file.readlines()
#
# for line in content:
#     file_name = line.strip() + ".txt"
#     full_path = os.path.join(file_path, file_name)
#
#     with open(full_path, 'r') as file:
#         X_unsplit = file.readlines()[window]
#         X_parts = X_unsplit.strip().split(",")
#         X.append(X_parts)
#         y.append(isDeliberate)
#
# return X, y
