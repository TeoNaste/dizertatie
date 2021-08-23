import tkinter as tk

from controllers.TrainerController import TrainerController
from utils.Utils import Utils


class SearchPage(tk.Toplevel):

    def __init__(self, controller: TrainerController, master=None):
        super().__init__(master=master)
        self.controller = controller
        self.utils = Utils()
        self.start()

    def start(self):
        self.title("Fake news detection - Search")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # create main frame
        main_frame = tk.Frame(self, width=400)
        main_frame.grid(column=0, row=0, sticky=tk.NSEW)
        main_frame.rowconfigure(0, weight=2)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        #create search bar frame
        top_frame = tk.Frame(self, bg="orange",width=400)
        top_frame.columnconfigure(0,weight=1)
        top_frame.grid(row=0,column=0,sticky=tk.NSEW)

        search_label = tk.Label(top_frame, text='Search', font='helvetica')
        search_label.grid(column=0, row=0, sticky=tk.EW,pady=10)

        search_entry = tk.Entry(top_frame)
        search_entry.place(height=100)
        search_entry.insert(tk.END, 'Search for a claim...')
        search_entry.grid(column=0,row=1,padx=30,pady=(5,10),sticky=tk.EW)

        options_button = tk.Button(top_frame, text="Show options")
        options_button.bind("<Button>", lambda e: print("Show options"))
        options_button.grid(column=0,row=2,sticky=tk.W,padx=10,pady=5)

        #middle frame - options frame
        options_frame = tk.Frame(self, bg="blue",height=60)
        options_frame.grid(column=0,row=1,sticky=tk.EW+tk.N,pady=10, padx=10)

        #bottom frame - recent searches
        bottom_frame = tk.Frame(self, bg="green",height=100)
        bottom_frame.grid(column=0, row=2, sticky=tk.EW + tk.S)