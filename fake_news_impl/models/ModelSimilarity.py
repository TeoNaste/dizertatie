import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dataLoaders import FileDataLoader

KNOWLEDGE_BASE_PATH = '../datasets/knowledge_base.csv'


class ModelSimilarity:

    def __init__(self, loader : FileDataLoader):
        self.__data_loader = loader
        self.__knowledge_base_claims = []
        self.__knowledge_base_labels = []
        self.__N = 5
        self.load_knowledge_base()

    def get_top_similar(self, sentence, label):
        """
        Returns an array with the top N most similar claims to the sentence
        :param sentence: array of tokens
        :param label: 0 = False, 1 = True
        :return: array of most similar tokenized claims from the knowledge base + their tokens
        """
        self.add_sentence_to_knowledge_base(sentence,label)
        similarity_matrix = self.compute_cosine_similarity(self.__knowledge_base_claims)

        #get similarity for the sentence aka last row appended with indexes to help find the claims
        sentence_similarity = list(enumerate(similarity_matrix[len(similarity_matrix)-1]))
        top_N_pairs = sorted(sentence_similarity, key=lambda x:x[1])[-self.__N:]
        #document index of the most similar documents
        top_N_index = list(reversed([i for i,v in top_N_pairs]))[1:]

        #get full text claims for knowledge base and return them as an array
        similar_claims = [self.__knowledge_base_claims[i] for i in top_N_index]
        similar_index = [self.__knowledge_base_labels[i] for i in top_N_index]

        return similar_claims,similar_index

    def compute_cosine_similarity(self, textlist):
        """
        Computes cosine similarity using sklearn
        :param textlist: text corpus, including the new sentence
        :return: similarity matrix
        """
        tfidf = TfidfVectorizer().fit_transform(textlist)
        return (tfidf * tfidf.T).toarray()

    def add_sentence_to_knowledge_base(self,sentence, label):
        """
        Adds new sentence and its label to the knowledge base corpus
        :param sentence: new sentence
        :param label: label for the new sentence
        """
        self.__knowledge_base_claims = np.append(sentence,self.__knowledge_base_claims)
        self.__knowledge_base_labels = np.append(label,self.__knowledge_base_labels)

    def set_N(self,n:int):
        """
        Set the number N of similar claims returned by the model (5 by default)
        :param n: new value
        """
        self.__N = n

    def load_knowledge_base(self):
        """
        Populates the claims and labels from the knowledge base corpus
        """
        self.__knowledge_base_claims,self.__knowledge_base_labels = self.__data_loader.load_dataset(KNOWLEDGE_BASE_PATH)