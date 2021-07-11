import csv
import numpy as np

from preprocessing.DataPreprocessor import DataPreprocessor


class FileDataLoader:

    def __init__(self, dataPreprocesor : DataPreprocessor):
        self.processor = dataPreprocesor
        self.folder_path = 'datasets/'

    def load_and_process_dataset(self,filename:str):
        """
        Reads a csv file and creates a dataset and labels after preprocessing the text and tags.
        :param filename: file that contains the data
        :return: dataset and labels as numpy arrays
        """
        dataset = []
        labels = []
        with open(self.folder_path+filename,'r',encoding="utf8") as file:
            reader = csv.reader(file)
            for i,row in enumerate(reader):
                if i == 0:
                    continue
                id = row[0]
                text = self.processor.preprocess_sample(row[1])
                explanation = self.processor.preprocess_sample(row[2])
                tag = self.processor.preprocess_tag(row[3])

                dataset.append([id,text,explanation])
                labels.append(tag)

        return np.array(dataset),np.array(labels)

    def load_dataset(self,filename:str):
        """
        Reads a csv file and creates a dataset.
        :param filename: file that contains processed data
        :return: dataset ad labels as numpy arrays
        """
        dataset = []
        labels = []
        with open(filename, 'r', encoding="utf8") as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                id = row[0]
                text = row[1]
                explanation = row[2]
                tag = row[3]

                dataset.append([id, text, explanation])
                labels.append(tag)

        return np.array(dataset), np.array(labels)

    def save_dataset(self,dataset,labels,filename:str, append:bool):
        """
        Save a dataset to a file
        :param dataset: data as numpy array
        :param labels: labels as numpy array
        :param filename: file to save data
        :param append: if set to true it appends data to file
        """
        action = 'w'
        if append:
            action = 'a'

        with open(self.folder_path+filename,action, newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            for i,row in enumerate(dataset):
                writer.writerow([row[0],row[1],row[2],labels[i]])

    def save_array(self,array,filename:str):
        """
        Saves an array structure to a file
        :param array: numpy array
        :param filename: the destionation file
        """

        with open(self.folder_path+filename,'w',newline='') as file:
            file.writelines(array)
