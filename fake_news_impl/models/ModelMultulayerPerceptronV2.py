from keras import regularizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


class ModelMultilayerPerceptronV2:

    def __init__(self, model_name:str):
        self.name = model_name

    def create_model(self, nr_features: int, activation: str, loss: str, layers: int,learning_rate:float, neurons: int):
        """
        Creats a model for a multi-layer perceptron neural network
        :param neurons: number of neurons on each layer
        :param learning_rate: number between 0-1
        :param layers: number of layers
        :param nr_features: number of features for the one input
        :param activation: name of the activation function
        :param loss: name of the loss function
        :return: keras model of a multi-layer perceptron neural network
        """
        model = Sequential(name=self.name)

        model.add(Dense(neurons, input_dim=nr_features, activation=activation))

        # Add hidden layers
        for layer in range(layers-1):
            # The input_dim argument creates the input layer with the right shape
            model.add(Dense(neurons, activation=activation))
            # Add dropout layer to reduce overfitting
            model.add(Dropout(0.2))

        #Output layer
        #Uses sigmoid because it is easier to map for the true / false labels
        #Applied l2 regularization
        model.add(Dense(1,activation='sigmoid',activity_regularizer=regularizers.l2(1e-5)))

        #Compile model
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

        return model
