import csv
from Config.persistanData import ClassifierVariant, MetricVariant, AppVariant, FoldSummaryMode

with open(r'todo.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    for classifier in ClassifierVariant:
        for metric in MetricVariant:
            for variant in AppVariant:
                for summary in FoldSummaryMode:
                    fields = [classifier, metric, variant, summary]
                    writer.writerow(fields)