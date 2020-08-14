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
import matplotlib

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

        self.t1_button = tk.Button(frame2, text="T1 scan", width=10, command=lambda: self.ask_for_file(self.t1_entry))
        self.t1_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.t1_entry = tk.Entry(frame2)
        self.t1_entry.pack(side=tk.LEFT, fill=tk.X, padx=5, expand=True)
        
        self.flair_button = tk.Button(frame2, text="FLAIR scan", width=10, command=lambda: self.ask_for_file(self.flair_entry))
        self.flair_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.flair_entry = tk.Entry(frame2)
        self.flair_entry.pack(side=tk.RIGHT, fill=tk.X, padx=5, expand=True)
        
        
        
        frame2p125 = Frame(self)
        frame2p125.pack(fill=tk.X)

        self.run_bianca_button = tk.Button(frame2p125, text="RUN BIANCA", width=40, command=self.run_bianca)
        self.run_bianca_button.pack(side=None, padx=5, pady=5)
        
        
        
        frame2p25 = Frame(self)
        frame2p25.pack(fill=tk.X)
        
        self.bg_scan = tk.StringVar()
        
        self.radio_t1 = tk.Radiobutton(frame2p25, text="T1", variable=self.bg_scan, value='t1', command=lambda: self.display_scan(self.bg_scan.get(), self.slice_slider.get()))
        self.radio_t1.pack(anchor=tk.W, padx=4, pady=3, side=tk.LEFT)
        
        self.radio_flair = tk.Radiobutton(frame2p25, text="FLAIR", variable=self.bg_scan, value='flair', command=lambda: self.display_scan(self.bg_scan.get(), self.slice_slider.get()))
        self.radio_flair.pack(anchor=tk.W, padx=4, pady=3, side=tk.LEFT)
        
        self.mask_on = tk.BooleanVar()
        
        self.mask_on_checkbox = tk.Checkbutton(frame2p25, text="Mask on", variable=self.mask_on, command=lambda: self.display_scan(self.bg_scan.get(), self.slice_slider.get()))
        self.mask_on_checkbox.pack(anchor=tk.W, padx=4, pady=3, side=tk.LEFT)
        
        alpha_label = tk.Label(frame2p25, text="Mask alpha", width=8, command=None)
        alpha_label.pack(side=tk.LEFT, padx=1, pady=3)
        
        self.alpha_entry = Entry(frame2p25)
        self.alpha_entry.pack(fill=tk.X, padx=4, expand=False, side=tk.LEFT)
        
        t1_max_label = tk.Label(frame2p25, text="T1 max", width=8)
        t1_max_label.pack(side=tk.LEFT, padx=1, pady=3)
        
        self.t1_max = Entry(frame2p25)
        self.t1_max.pack(fill=tk.X, padx=4, expand=False, side=tk.LEFT)
        
        flair_max_label = tk.Label(frame2p25, text="FLAIR max", width=8)
        flair_max_label.pack(side=tk.LEFT, padx=1, pady=3)
        
        self.flair_max = Entry(frame2p25)
        self.flair_max.pack(fill=tk.X, padx=4, expand=False, side=tk.LEFT)
        
        
        
        frame2p5 = Frame(self)
        frame2p5.pack(fill=tk.X)
        
        self.binarize_probability_mask_button = tk.Button(frame2p5, text="Binarize mask", width=10, command=self.binarize_probability_mask)
        self.binarize_probability_mask_button.pack(padx=2, pady=2, side=tk.LEFT, anchor=tk.S)
        
        self.binarize_slider = tk.Scale(frame2p5, from_=1, to=99, orient=tk.HORIZONTAL, command=lambda x: self.display_scan(self.bg_scan.get(), self.slice_slider.get()))
        self.binarize_slider.pack(fill=tk.X, padx=2, pady=2, expand=True)



        self.frame3 = Frame(self)
        self.frame3.pack(fill=tk.BOTH, expand=True)
        
        ### PLOT STUFF
        
        
        # adding the subplot 
        self.fig = plt.Figure(figsize = (6, 6), dpi = 100) 
        self.plot1 = self.fig.add_subplot(111) 
        self.plot1.axis('off')
        
        # PLOT STUFF
        
        slice_label = Label(self.frame3, text="Slice", width=1, wraplength=1)
        slice_label.pack(side=tk.RIGHT, padx=2, pady=2, anchor=tk.CENTER)
        
        self.slice_slider = tk.Scale(self.frame3, from_=10, to=0, orient=tk.VERTICAL, command=lambda x: self.display_scan(self.bg_scan.get(), self.slice_slider.get()))
        self.slice_slider.pack(side=tk.RIGHT, fill=tk.BOTH, padx=2, pady=2, expand=False)
        
        
        
        frame4 = Frame(self)
        frame4.pack(fill=None, expand=False)

        self.display_label = tk.Label(frame4, text="Step 0: initialization", width=30)
        self.display_label.pack(side=tk.BOTTOM, padx=5, pady=5)
        
        
        
        frame5 = Frame(self)
        frame5.pack(fill=tk.BOTH, expand=False)
        
        self.erode_button = tk.Button(frame5, text="ERODE", width=10, command=None)
        self.erode_button.pack(padx=2, pady=2, side=tk.LEFT)
        
        self.dilate_button = tk.Button(frame5, text="DILATE", width=10, command=None)
        self.dilate_button.pack(padx=2, pady=2, side=tk.LEFT)
        
        self.open_button = tk.Button(frame5, text="OPEN", width=10, command=None)
        self.open_button.pack(padx=2, pady=2, side=tk.LEFT)
        
        self.close_button = tk.Button(frame5, text="CLOSE", width=10, command=None)
        self.close_button.pack(padx=2, pady=2, side=tk.LEFT)
        

        
        self.save_mask_button = tk.Button(frame5, text="Save mask", width=12, command=None)
        self.save_mask_button.pack(padx=6, pady=2, side=tk.RIGHT)
        
        self.calculate_stats_button = tk.Button(frame5, text="Write statistics", width=12, command=None)
        self.calculate_stats_button.pack(padx=6, pady=2, side=tk.RIGHT)
        
        self.return_to_binarization_button = tk.Button(frame5, text="Return to binarization", width=20, command=self.return_to_binarization_cmd)
        self.return_to_binarization_button.pack(padx=6, pady=2, side=tk.RIGHT)
        
        
        # set up the overlay cmap
        
        viridis = matplotlib.cm.get_cmap('viridis', 256)
        newcolors1 = viridis(np.linspace(0, 1, 256))
        newcolors2 = viridis(np.linspace(0, 1, 256))
        
        red = np.array([255/255, 0/255, 0/255, 1])
        clear_blue = np.array([0/255, 0/255, 255/255, 0.05])
        maroon = np.array([128/255, 0/255, 0/255, 1])
        clear_navy = np.array([0/255, 0/255, 128/255, 0.05])
        
        blue = np.array([0/255, 0/255, 255/255, 1])
        navy = np.array([0/255, 0/255, 128/255, 1])
        
        newcolors1[:128, :] = clear_blue
        newcolors1[128:, :] = red
        
        self.probability_cmp = matplotlib.colors.ListedColormap(newcolors1)
        
        newcolors2[:128, :] = clear_navy
        newcolors2[128:, :] = blue
        
        self.binary_cmp = matplotlib.colors.ListedColormap(newcolors2)
        
        self.setup_stage(1)
        
        
    def return_to_binarization_cmd(self):
        self.current_overlay = self.probability_map
        self.setup_stage(2)
        self.display_scan(self.bg_scan.get(), self.slice_slider.get())
        
    def setup_stage(self, stage):
        """
        Current widgets:
            
        
        """
        #print(f'GOING TO STAGE {stage}')
        if stage == 1:
            self.display_label.config(text='Step 1: probability map generation')

            self.bianca_entry['state'] = tk.NORMAL
            self.t1_button['state'] = tk.NORMAL
            self.flair_button ['state']= tk.NORMAL
            self.t1_entry['state'] = tk.NORMAL
            self.flair_entry['state'] = tk.NORMAL
            self.run_bianca_button['state'] = tk.NORMAL
            
            self.radio_t1['state'] = tk.DISABLED
            self.mask_on_checkbox['state'] = tk.DISABLED
            self.radio_flair['state'] = tk.DISABLED
            self.alpha_entry['state'] = tk.DISABLED
            self.t1_max['state'] = tk.DISABLED
            self.flair_max['state'] = tk.DISABLED
            self.binarize_probability_mask_button['state'] = tk.DISABLED
            self.binarize_slider['state'] = tk.DISABLED
            self.slice_slider['state'] = tk.DISABLED
            
            self.erode_button['state'] = tk.DISABLED
            self.dilate_button['state'] = tk.DISABLED
            self.open_button['state'] = tk.DISABLED
            self.close_button['state'] = tk.DISABLED
            self.save_mask_button['state'] = tk.DISABLED
            self.calculate_stats_button['state'] = tk.DISABLED
            self.return_to_binarization_button['state'] = tk.DISABLED
            
        elif stage == 2:
            self.display_label.config(text='Step 2: probability map binarization')
            self.overlay_cmap = self.probability_cmp

            self.bianca_entry['state'] = tk.NORMAL
            self.t1_button['state'] = tk.NORMAL
            self.flair_button ['state']= tk.NORMAL
            self.t1_entry['state'] = tk.NORMAL
            self.flair_entry['state'] = tk.NORMAL
            self.run_bianca_button['state'] = tk.NORMAL
            
            self.radio_t1['state'] = tk.NORMAL
            self.mask_on_checkbox['state'] = tk.NORMAL
            self.radio_flair['state'] = tk.NORMAL
            self.alpha_entry['state'] = tk.NORMAL
            self.t1_max['state'] = tk.NORMAL
            self.flair_max['state'] = tk.NORMAL
            self.binarize_probability_mask_button['state'] = tk.NORMAL
            self.binarize_slider['state'] = tk.NORMAL
            self.slice_slider['state'] = tk.NORMAL
            
            self.erode_button['state'] = tk.DISABLED
            self.dilate_button['state'] = tk.DISABLED
            self.open_button['state'] = tk.DISABLED
            self.close_button['state'] = tk.DISABLED
            self.save_mask_button['state'] = tk.DISABLED
            self.calculate_stats_button['state'] = tk.DISABLED
            self.return_to_binarization_button['state'] = tk.DISABLED
            
        elif stage == 3:
            self.display_label.config(text='Step 3: binary map adjustment')
            self.overlay_cmap = self.binary_cmp
        
            self.bianca_entry['state'] = tk.NORMAL
            self.t1_button['state'] = tk.NORMAL
            self.flair_button ['state']= tk.NORMAL
            self.t1_entry['state'] = tk.NORMAL
            self.flair_entry['state'] = tk.NORMAL
            self.run_bianca_button['state'] = tk.NORMAL
            
            self.radio_t1['state'] = tk.NORMAL
            self.mask_on_checkbox['state'] = tk.NORMAL
            self.radio_flair['state'] = tk.NORMAL
            self.alpha_entry['state'] = tk.NORMAL
            self.t1_max['state'] = tk.NORMAL
            self.flair_max['state'] = tk.NORMAL
            self.binarize_probability_mask_button['state'] = tk.DISABLED
            self.binarize_slider['state'] = tk.DISABLED
            self.slice_slider['state'] = tk.NORMAL
            
            self.erode_button['state'] = tk.NORMAL
            self.dilate_button['state'] = tk.NORMAL
            self.open_button['state'] = tk.NORMAL
            self.close_button['state'] = tk.NORMAL
            self.save_mask_button['state'] = tk.NORMAL
            self.calculate_stats_button['state'] = tk.NORMAL
            self.return_to_binarization_button['state'] = tk.NORMAL

    
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
            m = 'Please specify both T1 and FLAIR inputs'
            self.popupmsg(m)
            raise Exception('Insufficient imaging input')
            
                
        self.setup_stage(2)
        
        self.set_output_folder()
        bianca_master_file = ugh.generate_bianca_master(self.output_folder, self.flair_entry.get(), self.t1_entry.get())
        
        self.binarize_slider.set(50)
        
        self.alpha_entry.insert(0,0.5)
        
        self.probability_map_file = os.path.join(self.output_folder, 'probability_map.nii.gz')
        ugh.execute_bianca(master=bianca_master_file, model=self.bianca_entry.get(), outname=self.probability_map_file)
        
        self.probability_map = ugh.read_nifti_radiological(self.probability_map_file)
        self.flair = ugh.read_nifti_radiological(self.flair_entry.get())
        self.t1 = ugh.read_nifti_radiological(self.t1_entry.get())
        
        self.radio_flair.select()
        
        self.slice_slider.configure(from_=self.flair.shape[2])
        self.slice_slider.set(int(self.flair.shape[2]/2)-1)
        
        self.flair_max.insert(0, round(self.flair.max()))
        self.t1_max.insert(0, round(self.t1.max()))
        
        self.current_overlay = self.probability_map

        #self.popupmsg('Files loaded. BIANCA successfully executed')
        
    
    def binarize_probability_mask(self):
        
        self.setup_stage(3)
        
        self.binary_mask = self.probability_map.copy()
        self.binary_mask = self.binary_mask >= self.binarize_slider.get()/100
        self.binary_mask = self.binary_mask.astype(int)
        
        self.current_overlay = self.binary_mask
        
        self.display_scan(self.bg_scan.get(), self.slice_slider.get())
    
    
    def display_scan(self, scan, sli, cmap=matplotlib.cm.gray):
  
        
        #print('BG called')
        
        if scan == 't1':
            scan = self.t1
            max_inten =  self.t1_max.get()
        elif scan == 'flair':
            scan = self.flair
            max_inten =  self.flair_max.get()
            
        try:
            self.canvas
        except AttributeError:
            self.canvas = FigureCanvasTkAgg(self.fig, master = self.frame3) 
            self.canvas.get_tk_widget().pack() 
        
        ax_slice = scan[:,:,sli]
      
        # plotting the graph
        
        try:
            self.basedisplay.set_data(ax_slice)
            self.basedisplay.set_clim(vmax=max_inten)
        except AttributeError:
            self.basedisplay = self.plot1.imshow(ax_slice, cmap=cmap, vmin=0, vmax=max_inten)
            
        self.display_overlay(sli)
        
        self.fig.tight_layout()
      
        # creating the Tkinter canvas 
        # containing the Matplotlib figure
        
        self.canvas.draw() 
      
        # placing the canvas on the Tkinter window 
      
        
      
        # creating the Matplotlib toolbar 
        #toolbar = NavigationToolbar2Tk(canvas, self.frame3) 
        #toolbar.update() 
      
        # placing the toolbar on the Tkinter window 
        #canvas.get_tk_widget().pack()
        
    def get_effective_alpha(self):
        
        boo = self.mask_on.get()
        
        if not boo:
            return 0
        else:
            return float(self.alpha_entry.get())
        
    def display_overlay(self, sli):
        
        cmap = self.overlay_cmap
        #print('Overlay called')
        
        mi = self.binarize_slider.get()/100
        al = self.get_effective_alpha()

        scan = self.current_overlay

        ax_slice = scan[:,:,sli]
        
        ax_slice = ax_slice >= self.binarize_slider.get()/100
        ax_slice = ax_slice.astype(int)
        
        #offset = matplotlib.colors.DivergingNorm(vmin=0, vcenter=mi, vmax=1)
        #ax_slice = offset(ax_slice)
        
        try:
            self.overlaydisplay.set_data(ax_slice)
            self.overlaydisplay.set_cmap(cmap)
            #self.overlaydisplay.set_clim(vmin=0, vmax=1)
            self.overlaydisplay.set_alpha(al)
        except AttributeError:
            self.overlaydisplay = self.plot1.imshow(ax_slice, cmap=cmap, vmin=0, vmax=1, alpha=al)

    
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
    root.geometry("1200x900+400+200")
    app = MainApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()