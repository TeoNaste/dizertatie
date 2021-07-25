import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dataLoaders import FileDataLoader

KNOWLEDGE_BASE_PATH = 'knowledge_base.csv'


class ModelSimilarity:

    def __init__(self, loader : FileDataLoader):
        self.__data_loader = loader
        self.__knowledge_base_claims = []
        self.__knowledge_base_labels = []

    def create_model(self, N = 5):
        if len(self.__knowledge_base_claims) == 0:
            self.load_knowledge_base()
        self.set_N(N)
        return self

    def get_top_similar(self, sentence, label):
        """
        Returns an array with the top N most similar claims to the sentence
        :param sentence: [id, claim, explanation]
        :param label: 0 = False, 1 = True
        :return: array of most similar tokenized claims from the knowledge base + their tokens
        """
        self.add_sentence_to_knowledge_base(sentence,label)
        similarity_matrix = self.compute_cosine_similarity([claim[1] for claim in self.__knowledge_base_claims])

        #get similarity for the sentence aka last row appended with indexes to help find the claims
        sentence_similarity = list(enumerate(similarity_matrix[len(similarity_matrix)-1]))
        top_N_pairs = sorted(sentence_similarity, key=lambda x:x[1])[-self.__N:]
        #document index of the most similar documents
        top_N_index = list(reversed([i for i,v in top_N_pairs]))[1:]
        top_N_sim_values = list(reversed([v for i,v in top_N_pairs]))[1:]

        #get full text claims for knowledge base and return them as an array
        similar_claims = [self.__knowledge_base_claims[i] for i in top_N_index]
        similar_labels = [self.__knowledge_base_labels[i] for i in top_N_index]

        return similar_claims,similar_labels,top_N_sim_values

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
        self.__knowledge_base_claims = np.concatenate((self.__knowledge_base_claims,sentence.reshape(1,len(sentence))),axis=0)
        self.__knowledge_base_labels = np.concatenate((self.__knowledge_base_labels,[label]),axis=0)

    def set_N(self,n:int):
        """
        Set the number N of similar claims returned by the model (5 by default)
        :param n: new value
        """
        self.__N = n

    def get_labels(self,indexes):
        """
        Return Label for entry in knowledge base
        :param indexes: list of indexes
        :return: labels : int[]
        """
        labels = []
        for index in indexes:
            labels.append(self.__knowledge_base_labels[index])
        return labels

    def load_knowledge_base(self):
        """
        Populates the claims and labels from the knowledge base corpus
        """
        self.__knowledge_base_claims,self.__knowledge_base_labels = self.__data_loader.load_dataset(KNOWLEDGE_BASE_PATH)

    def get_knowledge_base(self):
        return self.__knowledge_base_claims
