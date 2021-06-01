import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class DataPreprocessor:

    def __init__(self):
        self.__stop_words = set(stopwords.words('english'))
        self.__lemmatizer = WordNetLemmatizer()

    def preprocess_sample(self,sample):
        sample = self.__lower_case(sample)
        sample = self.__tokenize(sample)
        sample = self.__remove_stop_words(sample)
        sample = self.__lemmatize(sample)

        return sample

    def preprocess_tag(self,tag):
        tag = self.__lower_case(tag)
        if tag == 'true':
            return 1
        else:
            return 0

    def __lower_case(self, text):
        return text.lower()

    def __remove_stop_words(self, text):
        """Assumes text has been tokenized."""
        return [w for w in text if w not in self.__stop_words]

    def __tokenize(self, text):
        """Assumes text is lowercase."""
        matches = re.findall('(\d+.\d%)|(\w+)', text) # ignore punctuation unless they are part of a percentage
        tokenized = []
        for pair in matches:
            if pair[0] != '':
                tokenized.append(pair[0])
            else:
                tokenized.append(pair[1])

        return tokenized

    def __lemmatize(self, text):
        """Assumes text has been tokenized."""
        return [self.__lemmatizer.lemmatize(w) for w in text]
