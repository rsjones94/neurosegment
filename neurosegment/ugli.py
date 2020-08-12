#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UGLI (User-guided lesion identification) is a set of command-line tools and
related GUI that helps generates lesion masks from pre-trained BIANCA models
"""


from tkinter import Tk, Text, TOP, BOTH, X, N, LEFT, Button, END
from tkinter.ttk import Frame, Label, Entry
from tkinter.filedialog import askopenfilename


class Example(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.master.title("UGLI: user-guided lesion identification")
        self.pack(fill=BOTH, expand=True)

        frame1 = Frame(self)
        frame1.pack(fill=X)

        lbl1 = Button(frame1, text="BIANCA labels", width=10, command=self.set_entry_path)
        lbl1.pack(side=LEFT, padx=5, pady=5)

        self.bianca_entry = Entry(frame1)
        self.bianca_entry.pack(fill=X, padx=5, expand=True)

        frame2 = Frame(self)
        frame2.pack(fill=X)

        lbl2 = Button(frame2, text="Master folder", width=10)
        lbl2.pack(side=LEFT, padx=5, pady=5)

        entry2 = Entry(frame2)
        entry2.pack(fill=X, padx=5, expand=True)

        frame3 = Frame(self)
        frame3.pack(fill=BOTH, expand=True)

        lbl3 = Label(frame3, text="Display", width=10)
        lbl3.pack(side=LEFT, anchor=N, padx=5, pady=5)

        txt = Text(frame3)
        txt.pack(fill=BOTH, pady=5, padx=5, expand=True)
        
    
    def set_entry_path(self):
        filename = askopenfilename()
        self.bianca_entry.delete(0,END)
        self.bianca_entry.insert(0,filename)
        

def main():

    root = Tk()
    root.geometry("300x300+300+300")
    app = Example()
    root.mainloop()


if __name__ == '__main__':
    main()