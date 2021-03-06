import tkinter as tk

from controllers.ResultsController import ResultsController
from controllers.TrainerController import TrainerController
from utils.Utils import Utils


class TrainingPage(tk.Toplevel):

    def __init__(self, controller: TrainerController, results_controller: ResultsController, master=None):
        super().__init__(master=master)
        self.controller = controller
        self.results_controller = results_controller
        self.utils = Utils()
        self.last_name = tk.StringVar()
        self.last_acc = tk.StringVar()
        self.last_loss = tk.StringVar()
        self.last_date = tk.StringVar()
        self.start()

    def start(self):
        self.title("Fake news detection - Training")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # create main frame
        main_frame = tk.Frame(self, width=600)
        main_frame.grid(column=0, row=0, sticky=tk.NSEW)
        main_frame.rowconfigure(0, weight=2)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # top frame - preprocessing
        top_frame = tk.Frame(self, bg="yellow", width=100, height=100)
        top_frame.columnconfigure(1, weight=3)
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(2, weight=2)
        top_frame.grid(column=0, row=0, sticky=tk.EW + tk.N)

        preprocessing_label = tk.Label(top_frame, text='Choose dataset file', font='helvetica')
        preprocessing_label.grid(column=0, row=0, sticky=tk.W)

        error_message_lbl = tk.Label(top_frame, bg='#fff', fg='#f00')
        error_message_lbl.grid(column=2, row=1, sticky=tk.EW)

        file_entry = tk.Entry(top_frame, width=50)
        file_entry.insert(tk.END, 'processed_data.csv')
        file_entry.grid(column=0, row=1, sticky=tk.EW, padx=5)

        process_btn = tk.Button(top_frame, text="Preprocess")
        process_btn.bind("<Button>", lambda e: self.__preprocess_file(file_entry.get(), error_message_lbl))
        process_btn.grid(column=1, row=1, padx=5, sticky=tk.EW)

        # middle frame - settings and results
        middle_frame = tk.Frame(self, width=600, height=100)
        middle_frame.columnconfigure(0, weight=2)
        middle_frame.columnconfigure(1, weight=4)
        middle_frame.grid(column=0, row=1, sticky=tk.NSEW, pady=10, padx=10)

        middle_left_frame = tk.Frame(middle_frame)
        middle_left_frame.grid(column=0, row=1, sticky=tk.NSEW)

        middle_right_frame = tk.Frame(middle_frame,width=100)
        middle_right_frame.columnconfigure(0, weight=1)
        middle_right_frame.columnconfigure(1, weight=1)
        middle_right_frame.columnconfigure(2, weight=1)
        middle_right_frame.columnconfigure(3, weight=1)
        middle_right_frame.grid(column=1, row=1, sticky=tk.NSEW)

        # middle left size = training parameters
        title_label = tk.Label(middle_left_frame, text='Training', font='helvetica')
        title_label.grid(row=0, columnspan=2, sticky=tk.W + tk.N)

        models_options = self.utils.get_available_models()
        model_option = tk.StringVar(middle_left_frame)
        model_option.set(models_options[0])
        models_dropdown = tk.OptionMenu(middle_left_frame, model_option, *models_options)
        models_dropdown.grid(row=1, columnspan=2, sticky=tk.EW + tk.N)

        batch_label = tk.Label(middle_left_frame, text="Batch size: ")
        batch_label.grid(column=0, row=2, sticky=tk.W)

        batch_entry = tk.Entry(middle_left_frame)
        batch_entry.insert(tk.END, '500')
        batch_entry.grid(column=1, row=2, sticky=tk.EW, padx=5)

        epochs_label = tk.Label(middle_left_frame, text="Epochs no: ")
        epochs_label.grid(column=0, row=3, sticky=tk.W)

        epochs_entry = tk.Entry(middle_left_frame)
        epochs_entry.insert(tk.END, '90')
        epochs_entry.grid(column=1, row=3, sticky=tk.EW, padx=5)

        features_label = tk.Label(middle_left_frame, text="Vocab size: ")
        features_label.grid(column=0, row=4, sticky=tk.W)

        features_entry = tk.Entry(middle_left_frame)
        features_entry.insert(tk.END, '500')
        features_entry.grid(column=1, row=4, sticky=tk.EW, padx=5)

        layers_label = tk.Label(middle_left_frame, text="Layers no: ")
        layers_label.grid(column=0, row=5, sticky=tk.W)

        layers_entry = tk.Entry(middle_left_frame)
        layers_entry.insert(tk.END, '5')
        layers_entry.grid(column=1, row=5, sticky=tk.EW, padx=5)

        nerons_label = tk.Label(middle_left_frame, text="Neurons no: ")
        nerons_label.grid(column=0, row=6, sticky=tk.W)

        neurons_entry = tk.Entry(middle_left_frame)
        neurons_entry.insert(tk.END, '100')
        neurons_entry.grid(column=1, row=6, sticky=tk.EW, padx=5)

        learningrate_label = tk.Label(middle_left_frame, text="Learning rate: ")
        learningrate_label.grid(column=0, row=7, sticky=tk.W)

        learningrate_entry = tk.Entry(middle_left_frame)
        learningrate_entry.insert(tk.END, '0.01')
        learningrate_entry.grid(column=1, row=7, sticky=tk.EW, padx=5)

        activation_label = tk.Label(middle_left_frame, text="Activation: ")
        activation_label.grid(column=0, row=8, sticky=tk.W)

        activation_options = self.utils.get_activations()
        activation_option = tk.StringVar(middle_left_frame)
        activation_option.set(activation_options[0])
        activations_dropdown = tk.OptionMenu(middle_left_frame, activation_option, *activation_options)
        activations_dropdown.grid(column=1, row=8, sticky=tk.EW + tk.N)

        loss_label = tk.Label(middle_left_frame, text="Loss: ")
        loss_label.grid(column=0, row=9, sticky=tk.W)

        loss_options = self.utils.get_loss_list()
        loss_option = tk.StringVar(middle_left_frame)
        loss_option.set(loss_options[0])
        loss_dropdown = tk.OptionMenu(middle_left_frame, loss_option, *loss_options)
        loss_dropdown.grid(column=1, row=9, sticky=tk.EW + tk.N)

        embedding_label = tk.Label(middle_left_frame, text="Embedding dim: ")
        embedding_label.grid(column=0, row=10, sticky=tk.W)

        embedding_entry = tk.Entry(middle_left_frame)
        embedding_entry.insert(tk.END, '50')
        embedding_entry.grid(column=1, row=10, sticky=tk.EW, padx=5)

        message_label = tk.Label(middle_left_frame, bg='#fff', fg='#f00')
        message_label.grid(row=12, columnspan=2, pady=(5, 0))

        train_btn = tk.Button(middle_left_frame, text="Train")
        train_btn.bind("<Button>", lambda e: self.__call_train(
            model_option.get(),
            file_entry.get(),
            int(batch_entry.get()),
            int(epochs_entry.get()),
            int(features_entry.get()),
            int(layers_entry.get()),
            float(learningrate_entry.get()),
            int(neurons_entry.get()),
            activation_option.get(),
            loss_option.get(),
            message_label,
            int(embedding_entry.get())))
        train_btn.grid(row=11, columnspan=2, padx=5, sticky=tk.EW)

        # middle right side - results
        results_label = tk.Label(middle_right_frame, text="Last result", font='helvetica')
        results_label.grid(row=0, columnspan=4, sticky=tk.W + tk.NS)

        last_name_label = tk.Entry(middle_right_frame,textvariable=self.last_name)
        last_name_label.grid(row=1,column=0,sticky=tk.NSEW)

        last_acc_label = tk.Entry(middle_right_frame,textvariable=self.last_acc)
        last_acc_label.grid(row=1, column=1, sticky=tk.NSEW)

        last_loss_label = tk.Entry(middle_right_frame,textvariable=self.last_loss)
        last_loss_label.grid(row=1, column=2, sticky=tk.NSEW)

        self.__show_last_results()

        results_label = tk.Label(middle_right_frame, text="Best Results", font='helvetica')
        results_label.grid(row=2, columnspan=4, sticky=tk.W + tk.NS)

        # button frame - progress
        bottom_frame = tk.Frame(self, bg="green", width=100, height=100)
        bottom_frame.grid(column=0, row=2, sticky=tk.EW + tk.S, pady=10, padx=10)

    def __preprocess_file(self, filename: str, label):
        if len(filename) > 0:
            label.config(fg='green')
            label.config(text='File saved!')
            self.controller.create_processed_data(filename, "processed_data.csv")
        else:
            label.config(text="No file chosen!")

    def __call_train(self,
                     model_name: str,
                     filename: str,
                     batch_size: int,
                     epochs: int,
                     feature_no: int,
                     layers: int,
                     learning_rate: float,
                     neurons: int,
                     activation: str,
                     loss: str,
                     message_label,
                     embedding_size):
        if self.utils.is_valid_model(model_name):
            message_label.config(fg='green')
            message_label.config(text='Training started, please wait...')
            self.controller.train_on_model(self.utils.get_model_name(model_name), filename, batch_size, epochs,
                                           feature_no, layers, learning_rate, neurons, activation, loss,
                                           embedding_size)
            self.__show_last_results()
            message_label.config(text='Training finished.')
        else:
            message_label.config(text='Please choose a model')

    def __show_last_results(self):
        if self.results_controller.last_ranked is not None:
            self.last_name.set(self.results_controller.last_ranked.model)
            self.last_acc.set(self.results_controller.last_ranked.accuracy)
            self.last_loss.set(self.results_controller.last_ranked.loss)
            self.last_date.set(self.results_controller.last_ranked.create_date)
