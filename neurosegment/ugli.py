#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UGLI (User-guided lesion identification) is a set of command-line tools and
related GUI that helps generates lesion masks from pre-trained BIANCA models
"""

import os

from tkinter import Tk
import tkinter as tk
from tkinter.ttk import Frame, Label, Entry, Radiobutton
from tkinter.filedialog import askopenfilename, askdirectory

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import pandas as pd

import ugli_helpers as ugh

class MainApp(Frame):

    def __init__(self, the_root):
        super().__init__()

        self.the_root = the_root
        self.init_UI()
        

    def init_UI(self):

        self.master.title("UGLI: user-guided lesion identification")
        self.pack(fill=tk.BOTH, expand=True)

        frame1 = Frame(self)
        frame1.pack(fill=tk.X)

        bianca_button = tk.Button(frame1, text="BIANCA model", width=10, command=lambda: self.ask_for_file(self.bianca_entry))
        bianca_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.bianca_entry = Entry(frame1)
        self.bianca_entry.pack(fill=tk.X, padx=5, expand=True)
        self.bianca_entry.insert(0,self.find_default_model())

        frame2 = Frame(self)
        frame2.pack(fill=tk.X)

        masterfile_button = tk.Button(frame2, text="Master folder", width=10, command=lambda: self.ask_for_dir(self.masterfile_entry))
        masterfile_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.masterfile_entry = tk.Entry(frame2)
        self.masterfile_entry.pack(side=tk.RIGHT, fill=tk.X, padx=5, expand=True)
        
        frame2p25 = Frame(self)
        frame2p25.pack(fill=tk.X)
        
        v = tk.IntVar()
        
        self.radio_t1 = tk.Radiobutton(frame2p25, text="T1", variable=v, value=1)
        self.radio_t1.pack(anchor=tk.W, padx=3, pady=3, side=tk.LEFT)
        
        self.radio_flair = tk.Radiobutton(frame2p25, text="FLAIR", variable=v, value=2)
        self.radio_flair.pack(anchor=tk.W, padx=3, pady=3, side=tk.LEFT)
        
        alpha_button = tk.Button(frame2p25, text="Set mask alpha", width=8, command=lambda: self.set_mask_alpha(self.alpha_entry.get()))
        alpha_button.pack(side=tk.LEFT, padx=2, pady=3)
        
        self.alpha_entry = Entry(frame2p25)
        self.alpha_entry.pack(fill=tk.X, padx=5, expand=False, side=tk.LEFT)
        self.alpha_entry.insert(0,50)
        
        frame2p5 = Frame(self)
        frame2p5.pack(fill=tk.X)
        
        binarize_probability_mask_button = tk.Button(frame2p5, text="Binarize mask", width=10, command=lambda: self.plot_graph(frame3, ax, self.binarize_slider.get()))
        binarize_probability_mask_button.pack(padx=2, pady=2, side=tk.LEFT, anchor=tk.S)
        
        self.binarize_slider = tk.Scale(frame2p5, from_=0, to=100, orient=tk.HORIZONTAL)
        self.binarize_slider.pack(fill=tk.X, padx=2, pady=2, expand=True)

        frame3 = Frame(self)
        frame3.pack(fill=tk.BOTH, expand=True)

        #txt = Text(frame3)
        #txt.pack(side=LEFT, fill=BOTH, pady=2, padx=2, expand=True)
        
        figure = plt.Figure(figsize=(6,5), dpi=100)
        ax = figure.add_subplot(111)
        
        chart_type = FigureCanvasTkAgg(figure, frame3)
        chart_type.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.plot_graph(frame3, ax)
        
        # do plt stuff
        
        slice_label = Label(frame3, text="Slice", width=1, wraplength=1)
        slice_label.pack(side=tk.RIGHT, padx=2, pady=2, anchor=tk.CENTER)
        
        self.slice_slider = tk.Scale(frame3, from_=10, to=0, orient=tk.VERTICAL)
        self.slice_slider.pack(side=tk.RIGHT, fill=tk.BOTH, padx=2, pady=2, expand=False)
        
        frame4 = Frame(self)
        frame4.pack(fill=tk.BOTH, expand=False)

        self.display_label = tk.Label(frame4, text="Step 1: model generation", width=30)
        self.display_label.pack(side=tk.BOTTOM, padx=5, pady=5)
        
        #frame5 = Frame(self)
        #frame5.pack(fill=BOTH, expand=False)
    
    
    def ask_for_file(self, entry):
        filename = askopenfilename()
        entry.delete(0,tk.END)
        entry.insert(0,filename)
        
        
    def ask_for_dir(self, entry):
        filename = askdirectory()
        entry.delete(0,tk.END)
        entry.insert(0,filename)
        
    
    def find_default_model(self):
        script_folder = os.path.dirname(os.path.realpath(__file__))
        repo_folder = os.path.dirname(script_folder)
        default_bianca = os.path.join(repo_folder, 'bin', 'ugli_bianca_models', 'default_bianca_classifer')
        return default_bianca
    
    
    def set_mask_alpha(self, n):
        print(n)
        return n
    
    
    def plot_graph(self, frame, ax, pert=1):
        print(pert)
        ax.clear()
        ex = np.array([0,2,5,8])
        why = np.array([10,6,8,3]) * ((np.random.rand(4)-0.5)*0.01*pert)
        ax.plot(ex,why)
        ax.set_title('Wowee')
        
        
    def just_print(self):
        print('ey')


        

def main():

    root = Tk()
    root.geometry("700x700+300+300")
    app = MainApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()