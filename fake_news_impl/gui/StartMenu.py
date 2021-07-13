import tkinter as tk

from controllers.Controller import Controller
from gui.TrainingPage import TrainingPage


class StarMenu:

    def __init__(self, controller: Controller):
        self.controller = controller

    def start(self):
        window = tk.Tk()
        window.title("Fake news detection")

        #create main frame
        main_frame = tk.Frame(window)
        main_frame.grid(column=0,row=0)
        main_frame.columnconfigure(0,weight=1)
        main_frame.rowconfigure(0,weight=1)
        main_frame.pack()

        #create greeting message
        greeting = tk.Label(text="Welcome!")
        greeting.pack()

        #menu
        train_btn = tk.Button(window,text="Training")
        train_btn.bind("<Button>", lambda e: self.open_trainig_page(window))
        train_btn.pack()

        window.mainloop()

    def open_trainig_page(self,window):
        training_page = TrainingPage(self.controller,window)
        training_page.start()