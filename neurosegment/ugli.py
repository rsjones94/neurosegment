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

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

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

        t1_button = tk.Button(frame2, text="T1 scan", width=10, command=lambda: self.ask_for_file(self.t1_entry))
        t1_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.t1_entry = tk.Entry(frame2)
        self.t1_entry.pack(side=tk.LEFT, fill=tk.X, padx=5, expand=True)
        
        flair_button = tk.Button(frame2, text="FLAIR scan", width=10, command=lambda: self.ask_for_file(self.flair_entry))
        flair_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.flair_entry = tk.Entry(frame2)
        self.flair_entry.pack(side=tk.RIGHT, fill=tk.X, padx=5, expand=True)
        
        
        
        frame2p125 = Frame(self)
        frame2p125.pack(fill=tk.X)

        run_bianca_button = tk.Button(frame2p125, text="RUN BIANCA", width=40, command=self.run_bianca)
        run_bianca_button.pack(side=None, padx=5, pady=5)
        
        
        
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
        
        binarize_probability_mask_button = tk.Button(frame2p5, text="Binarize mask", width=10, command=None)
        binarize_probability_mask_button.pack(padx=2, pady=2, side=tk.LEFT, anchor=tk.S)
        
        self.binarize_slider = tk.Scale(frame2p5, from_=0, to=100, orient=tk.HORIZONTAL)
        self.binarize_slider.pack(fill=tk.X, padx=2, pady=2, expand=True)



        frame3 = Frame(self)
        frame3.pack(fill=tk.BOTH, expand=True)
        
        ### PLOT STUFF
        
        txt = tk.Text(frame3)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, pady=2, padx=2, expand=True)
        
        # PLOT STUFF
        
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
        
        
    def set_output_folder(self):
        where_t1 = self.t1_entry.get()
        parent = os.path.dirname(where_t1)
        out = os.path.join(parent, 'ugli')
        
        if os.path.exists(out):
            m = f'When you hit the RUN BIANCA button, UGLI generates a subfolder called "ugli" in the same directory as the specified T1 scan\n\nUGLI has detected that {out} already exists. Please remove or rename this folder'
            self.popupmsg(m)
            raise Exception(f'Folder {out} already exists. Please delete or rename folder')
        else:
            os.mkdir(out)
            self.output_folder = out
            
    def run_bianca(self):
        
        if '' in [self.t1_entry.get(), self.flair_entry.get()]:
            m = f'Please specify both T1 and FLAIR inputs'
            self.popupmsg(m)
            raise Exception(f'Insufficient imaging input')
        
        self.set_output_folder()
        bianca_master_file = ugh.generate_bianca_master(self.output_folder, self.flair_entry.get(), self.t1_entry.get())
        
        self.probability_map = os.path.join(self.output_folder, 'probability_map.nii.gz')
        self.probability_map = ugh.execute_bianca(master=bianca_master_file, model=self.bianca_entry.get(), outname=self.probability_map)
        
        self.popupmsg('BIANCA successfully executed')

    
    def popupmsg(self, msg, title='!!! ATTENTION !!!'):
        popup = tk.Tk()
        popup.wm_title(title)
        label = tk.Label(popup, text=msg)
        label.pack(side="top", fill="x", pady=10, padx=10)
        B1 = tk.Button(popup, text="Got it", command = popup.destroy)
        B1.pack()
        popup.mainloop()
        

def main():

    root = Tk()
    root.geometry("700x700+300+300")
    app = MainApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()