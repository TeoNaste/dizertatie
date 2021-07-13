import tkinter as tk

from controllers.Controller import Controller


class TrainingPage(tk.Toplevel):

    def __init__(self,controller:Controller,master=None):
        super().__init__(master=master)
        self.controller = controller
        self.start()

    def start(self):
        self.title("Fake news detection - Training")
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)

        # create main frame
        main_frame = tk.Frame(self,width=300)
        main_frame.grid(column=0,row=0,sticky=tk.E+tk.W+tk.N+tk.S)
        main_frame.rowconfigure(0, weight=2)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0,weight=1)

        #top frame - preprocessing
        top_frame = tk.Frame(self,bg="yellow",width=100,height=100)
        top_frame.columnconfigure(1,weight=3)
        top_frame.columnconfigure(0,weight=1)
        top_frame.columnconfigure(2,weight=2)
        top_frame.grid(column=0,row=0,sticky=tk.E+tk.W)

        preprocessing_label = tk.Label(top_frame,text='Choose dataset file',font='helvetica')
        preprocessing_label.grid(column=0,row=0,sticky=tk.W)

        error_message_lbl = tk.Label(top_frame,bg='#fff', fg='#f00')
        error_message_lbl.grid(column=2, row=1, sticky=tk.W)

        file_entry = tk.Entry(top_frame)
        file_entry.insert(tk.END,'csvpoynter.csv')
        file_entry.grid(column=0, row=1, sticky=tk.EW, padx=10)

        process_btn = tk.Button(top_frame,text="Preprocess")
        process_btn.bind("<Button>", lambda e: self.__preprocess_file(file_entry.get(),error_message_lbl))
        process_btn.grid(column=1,row=1,padx=10)

        #middle frame - settings and results
        middle_frame = tk.Frame(self,width=100,height=100)
        middle_frame.grid(column=0,row=1,sticky=tk.E+tk.W+tk.N+tk.S)

        #button frame - progress
        bottom_frame = tk.Frame(self,bg="green",width=100,height=100)
        bottom_frame.grid(column=0,row=2,sticky=tk.E+tk.W)

    def __preprocess_file(self,filename:str,label):
        if len(filename) > 0:
            self.controller.create_processed_data(filename,"processed_data.csv")
        else:
            label.config(text="No file chosen!")

