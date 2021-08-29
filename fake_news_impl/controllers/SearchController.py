from preprocessing.DataPreprocessor import DataPreprocessor
from trainers.ModelTrainer import ModelTrainer
from utils.SharedData import SharedData

DEFAULT_MODEL = 'mlps'
DEFAULT_DATASET = 'datasets/processed_data.csv'


class SearchController:

    def __init__(self, trainer: ModelTrainer, preprocessor: DataPreprocessor,shared_data: SharedData):
        self.trainer = trainer
        self.preprocessor = preprocessor
        self.shared_data = shared_data
        self.trainer.load_model(self.shared_data.best_model.filename)

    def predict(self,claim: str, model=None):

        if model is not None:
            self.trainer.load_model(model)

        #clean and vectorize claim
        input = self.shared_data.claim_to_bag_of_words(self.preprocessor.preprocess_sample(claim))
        prediction = self.trainer.model.predict_classes(input)


