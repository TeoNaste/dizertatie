import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class DataPreprocessor:

    def __init__(self):
        self.__stop_words = set(stopwords.words('english'))
        self.__lemmatizer = WordNetLemmatizer()

    def preprocess_sample(self,sample):
        """
        Processes on sample (claim) at a time
        :param sample: one claim of the dataset
        :return: processed sample as string
        """
        sample = self.__lower_case(sample)
        sample = self.tokenize(sample)
        sample = self.__remove_stop_words(sample)
        sample = self.__lemmatize(sample)

        processed_string = ' '
        return processed_string.join(sample)

    def preprocess_tag(self,tag:str):
        """
        Formats tags from strings to int for easier processing
        :param tag: the tag given as a string from the dataset
        :return: 1, if tag is True; otherwise 0
        """
        tag = self.__lower_case(tag)
        if tag == 'true':
            return 1
        else:
            return 0

    def __lower_case(self, text:str):
        """
        Formats text to lowercase
        :param text: text from dataset
        :return: text all letters in lowercase
        """
        return text.lower()

    def __remove_stop_words(self, text):
        """
        Removes stop words. Assumes text has been tokenized
        :param text: tokenized text, array of tokens
        :return: array of tokens without stopwords
        """
        return [w for w in text if w not in self.__stop_words]

    def tokenize(self, text:str):
        """
        Tokenizes a string
        :param text: string from the dataset
        :return: array of tokens
        """
        matches = re.findall('(\d+.\d%)|(\w+)', text) # ignore punctuation unless they are part of a percentage
        tokenized = []
        for pair in matches:
            if pair[0] != '':
                tokenized.append(pair[0])
            else:
                tokenized.append(pair[1])

        return tokenized

    def __lemmatize(self, text):
        """
        Uses the NLTK lemmatizer to lemmatize a text
        :param text: array of tokens, excluding stop words
        :return: array of lemmas
        """
        return [self.__lemmatizer.lemmatize(w) for w in text]

