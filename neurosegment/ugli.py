#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UGLI (User-guided lesion identification) is a set of command-line tools and
related GUI that helps generates lesion masks from pre-trained BIANCA models
"""

import os

from tkinter import Tk, Text, TOP, BOTH, X, Y, N, W, E, S, LEFT, Button, END, HORIZONTAL, Scale, VERTICAL, RIGHT, BOTTOM, CENTER, IntVar
from tkinter.ttk import Frame, Label, Entry, Radiobutton
from tkinter.filedialog import askopenfilename, askdirectory

import ugli_helpers as ugh

class MainApp(Frame):

    def __init__(self):
        super().__init__()

        self.init_UI()


    def init_UI(self):

        self.master.title("UGLI: user-guided lesion identification")
        self.pack(fill=BOTH, expand=True)

        frame1 = Frame(self)
        frame1.pack(fill=X)

        bianca_button = Button(frame1, text="BIANCA model", width=10, command=lambda: self.ask_for_file(self.bianca_entry))
        bianca_button.pack(side=LEFT, padx=5, pady=5)

        self.bianca_entry = Entry(frame1)
        self.bianca_entry.pack(fill=X, padx=5, expand=True)
        self.bianca_entry.insert(0,self.find_default_model())

        frame2 = Frame(self)
        frame2.pack(fill=X)

        masterfile_button = Button(frame2, text="Master folder", width=10, command=lambda: self.ask_for_dir(self.masterfile_entry))
        masterfile_button.pack(side=LEFT, padx=5, pady=5)

        self.masterfile_entry = Entry(frame2)
        self.masterfile_entry.pack(side=RIGHT, fill=X, padx=5, expand=True)
        
        frame2p25 = Frame(self)
        frame2p25.pack(fill=X)
        
        v = IntVar()
        
        self.radio_t1 = Radiobutton(frame2p25, text="T1", variable=v, value=1)
        self.radio_t1.pack(anchor=W, padx=3, pady=3, side=LEFT)
        
        self.radio_flair = Radiobutton(frame2p25, text="FLAIR", variable=v, value=2)
        self.radio_flair.pack(anchor=W, padx=3, pady=3, side=LEFT)
        
        alpha_button = Button(frame2p25, text="Set mask alpha", width=8, command=lambda: self.set_mask_alpha(self.alpha_entry.get()))
        alpha_button.pack(side=LEFT, padx=2, pady=3)
        
        self.alpha_entry = Entry(frame2p25)
        self.alpha_entry.pack(fill=X, padx=5, expand=False, side=LEFT)
        self.alpha_entry.insert(0,50)
        
        frame2p5 = Frame(self)
        frame2p5.pack(fill=X)
        
        binarize_probability_mask_button = Button(frame2p5, text="Binarize mask", width=10, command=None)
        binarize_probability_mask_button.pack(padx=2, pady=2, side=LEFT, anchor=S)
        
        self.binarize_slider = Scale(frame2p5, from_=0, to=100, orient=HORIZONTAL)
        self.binarize_slider.pack(fill=X, padx=2, pady=2, expand=True)

        frame3 = Frame(self)
        frame3.pack(fill=BOTH, expand=True)

        txt = Text(frame3)
        txt.pack(side=LEFT, fill=BOTH, pady=2, padx=2, expand=True)
        
        slice_label = Label(frame3, text="Slice", width=1, wraplength=1)
        slice_label.pack(side=RIGHT, padx=2, pady=2, anchor=CENTER)
        
        self.slice_slider = Scale(frame3, from_=10, to=0, orient=VERTICAL)
        self.slice_slider.pack(side=RIGHT, fill=BOTH, padx=2, pady=2, expand=False)
        
        frame4 = Frame(self)
        frame4.pack(fill=BOTH, expand=False)

        self.display_label = Label(frame4, text="Step 1: model generation", width=30)
        self.display_label.pack(side=BOTTOM, padx=5, pady=5)
        
        #frame5 = Frame(self)
        #frame5.pack(fill=BOTH, expand=False)
    
    
    def ask_for_file(self, entry):
        filename = askopenfilename()
        entry.delete(0,END)
        entry.insert(0,filename)
        
        
    def ask_for_dir(self, entry):
        filename = askdirectory()
        entry.delete(0,END)
        entry.insert(0,filename)
        
    
    def find_default_model(self):
        script_folder = os.path.dirname(os.path.realpath(__file__))
        repo_folder = os.path.dirname(script_folder)
        default_bianca = os.path.join(repo_folder, 'bin', 'ugli_bianca_models', 'default_bianca_classifer')
        return default_bianca
    
    def set_mask_alpha(self, n):
        print(n)
        return n
        

def main():

    root = Tk()
    root.geometry("700x700+300+300")
    app = MainApp()
    root.mainloop()


if __name__ == '__main__':
    main()