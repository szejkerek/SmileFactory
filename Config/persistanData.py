from enum import Enum
class AppVariant(Enum):
    Window_Pick_One = 1
    Window_Range = 2

class ClassifierVariant(Enum):
    RandomForest = 1
    DecisionTree = 2
    SVM = 3
    KNN = 4
    NeuralNet = 5
    NaiveBayes = 6

DataFolder = 'Resources\\UvA-NEMO\\features\\cross-AU window25'
FoldsPath = 'Resources\\list'