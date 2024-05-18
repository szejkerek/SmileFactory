from Config.persistanData import AppVariant, FoldSummaryMode, MetricVariant, ClassifierVariant

chosenClassifiers = [ClassifierVariant.RandomForest]
accuracyVariant = FoldSummaryMode.Average
metricVariant = MetricVariant.F1

mode = AppVariant.Window_Range
WindowMax = 3 #If window number
WindowSections = 3 #If Window_Pick_One or Window_Range

