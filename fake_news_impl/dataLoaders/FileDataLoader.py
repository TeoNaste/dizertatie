import csv

from preprocessing.DataPreprocessor import DataPreprocessor


class FileDataLoader:

    def __init__(self, dataPreprocesor : DataPreprocessor):
        self.processor = dataPreprocesor

    def load_and_process_dataset(self,filename):
        """Reads a csv file and creates a dataset after preprocessing the text and tags"""
        
        dataset = []
        with open(filename,'r',encoding="utf8") as file:
            reader = csv.reader(file)
            for i,row in enumerate(reader):
                if i == 0:
                    continue
                id = row[0]
                text = self.processor.preprocess_sample(row[1])
                explanation = self.processor.preprocess_sample(row[2])
                tag = self.processor.preprocess_tag(row[3])

                dataset.append([id,text,explanation,tag])
        return dataset
