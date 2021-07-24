from keras.utils import plot_model
import matplotlib.pyplot as plt


class ModelTrainer:

    def __init__(self,batch_size: int, epochs: int, dataset, labels, dataset_test, labels_test, model):
        """
        Initializes a trainer for a model
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :param dataset: Array of training samples
        :param labels: Array of labels for the training dataset
        :param dataset_test: Array of testing samples
        :param labels_test: Array of labels for the testing dataset
        :param model: the keras model for which the trainer is for
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = dataset
        self.labels = labels
        self.dataset_test = dataset_test
        self.labels_test = labels_test
        self.model = model
        self.__save_folder = "/savedModels/"

    def train(self):
        """
        Trains the keras model on the training dataset, then evaluates the model on the testing dataset
        """
        #Train the keras model
        history = self.model.fit(self.dataset,self.labels, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.dataset_test,self.labels_test))

        #Evaluate the keras model
        results = self.model.evaluate(self.dataset_test,self.labels_test)
        print("test loss, test acc: ", results)

        self.__plot_performance(history)
        #TODO: save results

    def save_model(self):
        """
        Saves a model after it's been trained
        """

        #Save model
        model_json = self.model.to_json()
        with open(self.__save_folder+self.model.name+'/model.json', "w") as json_fle:
            json_fle.write(model_json)

        #Save weights to HDF5
        self.model.save_weights(self.__save_folder+self.model.name+'/model.h5')

    def load_model(self, model_name: str):
        """
        Reads model from file
        """
        json_file = open(self.__save_folder+model_name+'/model.json')
        self.model = json_file.read()
        json_file.close()

        #read weights from file
        self.model.load_weights(self.__save_folder+model_name+'/model.h5')

    def __plot_performance(self, history):
        """
        Generates plot for loss and accuracy of the trained model
        """
        #Plot history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        #Plot history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def summarize_model(self, graph: bool):
        """
        Prints the summary of the model for the trainer
        :param graph: if set to True, it also computes and saves the model's graph on disk
        """
        if self.model is None:
            print("Trainer has no model defined")
            return

        print(self.model.summary())
        if graph:
            plot_model(self.model, to_file=self.__save_folder+self.model.name+'/plot.png', show_shapes= True, show_layer_names=True)