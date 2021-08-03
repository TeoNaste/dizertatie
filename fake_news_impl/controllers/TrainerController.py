import numpy as np
import threading
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from dataLoaders.FileDataLoader import FileDataLoader
from sklearn.model_selection import train_test_split

from models.ModelLstm import ModelLSTM
from models.ModelMultulayerPerceptronV2 import ModelMultilayerPerceptronV2
from models.ModelSimilarity import ModelSimilarity
from trainers.ModelTrainer import ModelTrainer

PREPROCESSED_PATH = 'processed_data.csv'
GLOVE_INPUT_PATH = '../../glove/glove.42B.300d.txt'
GLOVE_OUTPUT_PATH = '../../glove/glove.42B.300d.txt.word2vec'


class TrainerController:

    def __init__(self, data_loader: FileDataLoader):
        self.data_loader = data_loader
        self.vocab = []
        self.embeddings_index = None
        self.embedding_thread = threading.Thread(target=self.prepare_training)
        # self.embedding_thread.start()

    def create_processed_data(self, filename:str, filename_processed:str):
        """
        Preprocesses data and saves in a csv file.
        :param filename: file of raw data
        :param filename_processed: file to save the processed data
        """
        dataset, labels = self.data_loader.load_and_process_dataset(filename)

        self.data_loader.save_dataset(dataset,labels,filename_processed,False)

    def train_on_model(self,model_name:str, preprocessed_filename:str,batch_size:int,epochs:int,vocab_size:int,layers:int,learning_rate:float,neurons:int,activation:str,loss:str, embedding_size:int = 0):
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
            if embedding_size == 0:
                # vectorize
                X_train = self.text_to_bag_of_words(self.vocab, X_train)
                X_test = self.text_to_bag_of_words(self.vocab, X_test)

                # reshape as 3d
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                model = ModelLSTM(model_name).create_model(X_train.shape[1],activation,loss,layers,learning_rate,neurons)
            else:
                indexed_data, embedding_matrix = self.text_to_word_embeddings(self.vocab,embedding_size,dataset)
                model = ModelLSTM(model_name).create_model(X_train.shape[1], activation, loss, layers, learning_rate,
                                                           neurons,embedding_size,vocab_size, embedding_matrix)
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
        Vectorize a text data into a BoW with TF
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

    def text_to_word_embeddings(self,vocab,embedding_dim, data):
        """
        Vectorize a text data using GloVe word embeddings
        :param embedding_dim: dimension of embeddings
        :param vocab_size: size of vocabulary
        :param data: dataset
        :return: padded and indexed data, embedding_matrix
        """
        indexed_data,word_index = self.text_to_indexes(len(vocab),data)
        indexed_data = self.__add_padding(indexed_data, max(len(i) for i in vocab))

        embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))

        # self.embedding_thread.join()

        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # for word, i in word_index.items():
        #     embedding_vector = self.embeddinsgs_index.get(word)
        #     if embedding_vector is not None:
        #         embedding_matrix[i] = embedding_vector[:embedding_dim]

        return indexed_data, embedding_matrix

    def text_to_indexes(self, vocab_size, data):
        text_data = [item[1] for item in data]
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(text_data)

        indexed = tokenizer.texts_to_sequences(text_data)
        return indexed, tokenizer.word_index

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

    def prepare_training(self):
        print("Loading embeddings started...")
        self.embeddings_index = self.data_loader.load_embedding_indexes()
        print("Embeddings are loaded.")

    def __flatten_list(self,list):
        return [[item for sublist in list for item in sublist]]

    def __add_padding(self,data,max_len):
        return pad_sequences(data,maxlen=max_len, padding='post')

    def __convert_to_dict(self,list):
        return {item[0] : item[1] for item in list}
