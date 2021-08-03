import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from dataLoaders.FileDataLoader import FileDataLoader
from sklearn.model_selection import train_test_split

from models.ModelLstm import ModelLSTM
from models.ModelMultulayerPerceptronV2 import ModelMultilayerPerceptronV2
from models.ModelSimilarity import ModelSimilarity
from trainers.ModelTrainer import ModelTrainer

PREPROCESSED_PATH = 'processed_data.csv'
GLOVE_INPUT_PATH = '../../glove/glove.42B.300d.txt'
GLOVE_OUTPUT_PATH = '../../glove/glove.42B.300d.txt.word2vec'


class Controller:

    def __init__(self, data_loader: FileDataLoader):
        self.data_loader = data_loader
        self.vocab = []

    def create_processed_data(self, filename:str, filename_processed:str):
        """
        Preprocesses data and saves in a csv file.
        :param filename: file of raw data
        :param filename_processed: file to save the processed data
        """
        dataset, labels = self.data_loader.load_and_process_dataset(filename)

        self.data_loader.save_dataset(dataset,labels,filename_processed,False)

    def train_on_model(self,model_name:str, preprocessed_filename:str,batch_size:int,epochs:int,vocab_size:int,layers:int,learning_rate:float,neurons:int,activation:str,loss:str):
        #load preprocessed data
        dataset, labels = self.data_loader.load_dataset(preprocessed_filename)
        self.vocab = self.compute_most_frequent_words_vocabulary(dataset,vocab_size)

        #split into 80/20%
        X_train, X_test, y_train, y_test = self.split_data(dataset,labels)

        if model_name == 'mlp':
            # vectorize
            X_train = self.text_to_bag_of_words(self.vocab, X_train)
            X_test = self.text_to_bag_of_words(self.vocab, X_test)

            model = ModelMultilayerPerceptronV2(model_name).create_model(X_train.shape[1],activation,loss,layers,learning_rate,neurons)
            trainer = ModelTrainer(batch_size,epochs, X_train, y_train, X_test, y_test,model)
            trainer.train()

        if model_name == 'mlps':
            model_sim = ModelSimilarity(self.data_loader).create_model(2)
            similarity_dataset = self.create_dataset_for_similarity_models(dataset,labels,model_sim)

            # split into 80/20%
            X_train, X_test, y_train, y_test = self.split_data(similarity_dataset, labels)

            model = ModelMultilayerPerceptronV2(model_name).create_model(X_train.shape[1], activation, loss,layers,learning_rate,neurons)
            trainer = ModelTrainer(batch_size, epochs, X_train, y_train, X_test, y_test, model)
            trainer.train()

        if model_name == 'lstm':
            # vectorize
            X_train = self.text_to_bag_of_words(self.vocab, X_train)
            X_test = self.text_to_bag_of_words(self.vocab, X_test)

            #reshape as 3d
            X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
            X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

            model = ModelLSTM(model_name).create_model(batch_size,X_train.shape[1],X_train.shape[2],activation,loss,layers,learning_rate,neurons)
            trainer = ModelTrainer(batch_size,epochs,X_train, y_train, X_test, y_test,model)
            trainer.train()

    def compute_most_frequent_words_vocabulary(self, dataset,n:int):
        """
        Creates a vocabulary of the most frequent N words in a dataset and saves it to a file
        :param dataset: dataset as a numpy array
        :param n: number of words to be used in the vocabulary
        :return: vocabulary as dictionary
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

    def text_to_bag_of_words(self, vocab,data):
        """
        Vectorize a text data into a BoW with either TF (default) or TF-IDF index
        :param vocab: vocabulary used for the BoW as a numpy array
        :param data: array of text
        :return:
        """
        tf_vectors = []
        for entry in data:
            vector = []
            for word in vocab:
                vector.append(entry[1].split().count(word))
            tf_vectors.append(vector)
        return np.array(tf_vectors)

    def text_to_word_embeddings(self,vocab_size,data):
        #convert glove file to word2vec format
        glove2word2vec(GLOVE_INPUT_PATH, GLOVE_OUTPUT_PATH)
        
        model = KeyedVectors.load_word2vec_format(GLOVE_OUTPUT_PATH,binary=False)

    def create_dataset_for_similarity_models(self, data,labels,model_sim):
        all_similar_claims = []
        all_cosine_sim = []
        all_similar_labels = []

        for i,entry in enumerate(data):
            sim_claim,sim_labels, sim_values = model_sim.get_top_similar(entry,labels[i])
            all_similar_claims.append(sim_claim[0])
            all_similar_labels.append(sim_labels)
            all_cosine_sim.append(sim_values)

        #vectorize text data
        vect_data = self.text_to_bag_of_words(self.vocab, np.array(data))
        vect_similar = self.text_to_bag_of_words(self.vocab, all_similar_claims)

        #concatenate vectros of claim, cosine similarity between them and label of the most similar claim
        result1 = np.concatenate((vect_data,vect_similar),axis=1)
        result2 = np.concatenate((result1,np.array(self.__flatten_list(all_cosine_sim)).T),axis=1)
        result3 = np.concatenate((result2,np.array(self.__flatten_list(all_similar_labels)).T),axis=1)

        return result3

    def __flatten_list(self,list):
        return [[item for sublist in list for item in sublist]]



