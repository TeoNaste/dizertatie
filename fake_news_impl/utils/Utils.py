class Utils:

    def __init__(self):
        self.__models = {
            'mlp': 'Multilayer Perceptron'
        }

    def get_available_models(self):
        return self.__models