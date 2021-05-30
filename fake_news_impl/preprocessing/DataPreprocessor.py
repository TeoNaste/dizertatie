import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class DataPreprocessor:

    def __init__(self):
        self.__stop_words = set(stopwords.words('english'))
        self.__lemmatizer = WordNetLemmatizer()
        self.__replace = [
            (r'[^\w\s\d+\.\d+%]', ''),  # remove punctuation, but keep percentage
            (' +', ' '),  # remove duplicate spaces
            (r'\((.*)\)', r'\g<1>')  # remove parenthesis
        ]

    def lower_case(self, text):
        return text.lower()

    def remove_stop_words(self, text):
        """Assumes text has been tokenized."""
        return [w for w in text if w not in self.__stop_words]

    def tokenize(self, text):
        for old, new in self.__replace:
             text = re.sub(old, new, text)

        return text.split()

    def lemmatize(self, text):
        return [self.__lemmatizer.lemmatize(w) for w in text]

dp = DataPreprocessor()
text = "These masks have / negative: (impacts) on 68.8% of. the? children in 2002."
# text = dp.tokenize(text)
# text = dp.remove_stop_words(text)
print(re.sub(r'[^\w\s\d+\.\d%]', '',text))