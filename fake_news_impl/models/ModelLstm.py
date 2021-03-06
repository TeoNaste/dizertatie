from keras import Sequential, regularizers
from keras.layers import LSTM, Dense, Dropout, Embedding, Flatten, TimeDistributed, Bidirectional
from keras.optimizers import Adam


class ModelLSTM:

    def __init__(self, model_name:str):
        self.name = model_name

    def create_model(self, timesteps:int, activation: str, loss: str, layers: int,learning_rate:float, neurons: int, embedding_size: int = 0, vocab_size: int = 0):
        model = Sequential()
        model.name = self.name

        if embedding_size > 0:
            model.add(Embedding(input_dim=vocab_size,output_dim=embedding_size, input_length=timesteps))
            model.add(Dropout(0.4))

        model.add(LSTM(neurons, input_shape=(timesteps,1),return_sequences=True))
        model.add(Dropout(0.4))

        #Add the LSTM layers
        for layer in range(layers-2):
            model.add(LSTM(neurons, return_sequences=True))
            model.add(Dropout(0.4))

        model.add(LSTM(neurons))
        model.add(Dropout(0.4))

        # Output layer
        # Uses sigmoid because it is easier to map for the true / false labels
        # Applied l2 regularization
        model.add(Dense(1, activation=activation, activity_regularizer=regularizers.l2(1e-5)))

        # Compile model
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

        return model
