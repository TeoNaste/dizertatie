import numpy as np

from dataLoaders.FileDataLoader import FileDataLoader
from sklearn.model_selection import train_test_split

from models.ModelMultulayerPerceptronV2 import ModelMultilayerPerceptronV2
from trainers.ModelTrainer import ModelTrainer


class Controller:

    def __init__(self, data_loader: FileDataLoader):
        self.data_loader = data_loader

    def create_processed_data(self, filename:str, filename_processed:str):
        """
        Preprocesses data and saves in a csv file.
        :param filename: file of raw data
        :param filename_processed: file to save the processed data
        """
        dataset, labels = self.data_loader.load_and_process_dataset(filename)

        self.data_loader.save_dataset(dataset,labels,filename_processed,False)

    def train_on_model(self,model_name:str, preprocessed_filename:str,batch_size:int,epochs:int,n:int,activation:str,loss:str):
        #load preprocessed data
        dataset, labels = self.data_loader.load_dataset(preprocessed_filename)
        vocab = self.compute_most_frequent_words_vocabulary(dataset,n)

        #split into 80/20%
        X_train, X_test, y_train, y_test = self.split_data(dataset,labels)

        #vectorize
        X_train = self.text_to_bag_of_words(vocab,X_train)
        X_test = self.text_to_bag_of_words(vocab,X_test)

        if model_name == 'mlp':
            model = ModelMultilayerPerceptronV2(model_name).create_model(n,activation,loss)
            trainer = ModelTrainer(batch_size,epochs, X_train, y_train, X_test, y_test,model)
            trainer.train()


    def compute_most_frequent_words_vocabulary(self, dataset ,n:int):
        """
        Creates a vocabulary of the most frequent N words in a dataset and saves it to a file
        :param dataset: dataset as a numpy array
        :param n: number of words to be used in the vocabulary
        :return: vocabulary as dictionary ??
        """
        vocab = {}
        for entry in dataset:
            for word in entry[1].split():
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

        vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
        return [k for k in list(vocab.keys())[:n]]

    def split_data(self,dataset,labels):
        return train_test_split(dataset,labels, shuffle=True, test_size=0.2,random_state=11)

    # def text_to_word_embeddings(self):

    def text_to_bag_of_words(self, vocab,data, use_tfidf = False):
        """
        Vectorize a text data into a BoW with either TF (default) or TF-IDF index
        :param vocab: vocabulary used for the BoW as a numpy array
        :param data: array of text
        :param use_tfidf: mentions if either TF-IDF should be used
        :return:
        """
        tf_vectors = []
        for entry in data:
            vector = []
            for word in vocab:
                vector.append(entry[1].split().count(word))
            tf_vectors.append(vector)
        return np.array(tf_vectors)