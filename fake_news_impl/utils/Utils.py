class Utils:

    def __init__(self):
        self.__models = {
            '-- Choose a model to train --': 'none',
            'Multilayer Perceptron':'mlp',
            'MLP with similarity': 'mlps'
        }
        self.__activations = [
            'relu',
            'sigmoid'
        ]
        self.__loss = [
            'binary_crossentropy',
            'hinge',
            'mean_squared_error'
        ]

    def get_available_models(self):
        return list(self.__models.keys())

    def is_valid_model(self,model_display:str):
        return self.__models.get(model_display) != 'none'

    def get_model_name(self,model_display:str):
        return self.__models.get(model_display)

    def get_activations(self):
        return self.__activations

    def get_loss_list(self):
        return  self.__loss
