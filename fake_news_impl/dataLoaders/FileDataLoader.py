import csv

import numpy as np
from gensim.models import  KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from preprocessing.DataPreprocessor import DataPreprocessor

KNOWLEDGE_BASE_PATH = 'knowledge_base.csv'
RESULTS_PATH = 'results.txt'

class FileDataLoader:

    def __init__(self, dataPreprocessor : DataPreprocessor):
        self.processor = dataPreprocessor
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

        return self.shuffle_and_save_data(KNOWLEDGE_BASE_PATH,0.2,dataset,labels)

    def load_dataset(self,filename:str):
        """
        Reads a csv file and creates a dataset.
        :param filename: file that contains processed data
        :return: dataset ad labels as numpy arrays
        """
        dataset = []
        labels = []
        with open(self.folder_path+filename, 'r') as file:
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

    def save_dataset(self,dataset,labels,filename:str, append:bool = False):
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

    def shuffle_and_save_data(self,filename:str,N:float,dataset,labels):
        """
        Shuffles data rows and saves a percentage to a given file
        :param filename: name of the save file
        :param N: percetange to save (0-1)
        """
        dataset,labels = self.__shuffle_in_unison(dataset,labels)
        slice_index = int(len(dataset) * N)
        self.save_dataset(dataset[:slice_index],labels[:slice_index],filename)

        return np.array(dataset[slice_index+1:]),np.array(labels[slice_index+1:])

    def save_array(self,array,filename:str):
        """
        Saves an array structure to a file
        :param array: numpy array
        :param filename: the destionation file
        """

        with open(self.folder_path+filename,'w',newline='') as file:
            file.writelines(array)

    def load_embedding_indexes(self):
        word2vec_output_file = self.folder_path+'glove.42B.300d.word2vec'

        glove2word2vec(self.folder_path+'glove.6B.50d.txt', word2vec_output_file)

        model = self.__load_glove_model(word2vec_output_file)

        return model

    def load_results(self):
        with open(self.folder_path+RESULTS_PATH,'r') as file:
            reader = csv.reader(file)

    def __shuffle_in_unison(self,claims,labels):
        rng_state = np.random.get_state()
        np.random.shuffle(claims)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        return claims,labels

    def __load_glove_model(self, glove_file):
        return KeyedVectors.load_word2vec_format(glove_file, binary=False)
