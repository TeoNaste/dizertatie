import tkinter as tk
import asyncio

from controllers.TrainerController import TrainerController
from gui.SearchPage import SearchPage
from gui.TrainingPage import TrainingPage


class StarMenu:

    def __init__(self, controller: TrainerController):
        self.controller = controller

    def start(self):
        window = tk.Tk()
        window.title("Fake news detection")
        window.rowconfigure(0,weight=1)
        window.columnconfigure(0,weight=1)
        window.geometry("400x200")

        #create main frame
        main_frame = tk.Frame(window,height=200,width=400,bg='violet')
        window.columnconfigure(0, weight=1)
        main_frame.grid(column=0, row=0,sticky=tk.NSEW)

        #create greeting message
        greeting = tk.Label(window,text="Welcome!",font=("Courier", 35))
        greeting.grid(row=0,column=0)

        #menu
        train_btn = tk.Button(window,text="Training",width=30)
        train_btn.bind("<Button>", lambda e: self.open_trainig_page(window))
        train_btn.grid(row=1,column=0,sticky=tk.EW,columnspan=1,pady=(5,5),padx=(20,20))

        search_btn = tk.Button(window, text="Search",width=30 )
        search_btn.bind("<Button>", lambda e: self.open_search_page(window))
        search_btn.grid(row=2,column=0,sticky=tk.EW,columnspan=1,pady=(5,5),padx=(20,20))

        window.mainloop()

    def open_trainig_page(self,window):
        training_page = TrainingPage(self.controller,window)
        training_page.start()

    def open_search_page(self,window):
        search_page = SearchPage(self.controller,window)
        search_page.start()
