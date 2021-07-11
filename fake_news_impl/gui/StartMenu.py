import tkinter as tk


class StarMenu:

    def start(self):
        window = tk.Tk()
        window.title("Fake news detection")
        window.mainloop()

    def ensemble_window(self,window):
        #Add greeting label
        greeting = tk.Label(text="Welcome!")
        greeting.pack()