class SharedData:
    def __init__(self):
        self.best_model = None
        self.embeddings = None
        self.vocab = []
        self.dateset = []

    def claim_to_bag_of_words(self,claim:str):
        """
        Generates bag of words for one claim
        :param claim: cleaned string
        :return: vector count
        """
        vector = []
        for word in self.vocab:
            vector.append(claim.split().count(word))

        return vector

