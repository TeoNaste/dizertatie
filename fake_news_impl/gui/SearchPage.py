import tkinter as tk

from controllers.TrainerController import TrainerController
from utils.Utils import Utils


class PredictPage(tk.Toplevel):

    def __init__(self, controller: TrainerController, master=None):
        super().__init__(master=master)
        self.controller = controller
        self.utils = Utils()
        self.start()

    def start(self):
        self.title("Fake news detection - Search")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)