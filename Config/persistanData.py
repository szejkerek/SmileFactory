from enum import Enum
class AppVariant(Enum):
    Window_Number = 1
    Window_Pick_One = 2
    Window_Range = 3
class FoldSumaryMode(Enum):
    Average = 1
    Max = 2
    Min = 3
class MetricVariant(Enum):
    Accuracy = 1
    Precision = 2
    Recall = 3
    F1 = 4
    ROC = 5
    LogLoss = 6
DataFolder = 'Resources\\UvA-NEMO\\features\\cross-AU window25'
FoldsPath = 'Resources\\list'