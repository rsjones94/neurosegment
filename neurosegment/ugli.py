#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UGLI (User-guided lesion identification) is a set of command-line tools and
related GUI that helps generates lesion masks from pre-trained BIANCA models
"""

import os
import sys

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
import scipy.ndimage.morphology as mor
import nibabel as nib

import ugli_helpers as ugh
import gbs




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
        


        frame2p0625 = Frame(self)
        frame2p0625.pack(fill=tk.X)
        
        self.trans_button = tk.Button(frame2p0625, text="Transf. matrix", width=10, command=lambda: self.ask_for_file(self.trans_entry))
        self.trans_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.trans_entry = tk.Entry(frame2p0625)
        self.trans_entry.pack(side=tk.LEFT, fill=tk.X, padx=5, expand=True)
        
        
        
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
        
        alpha_label = tk.Label(frame2p25, text="Mask alpha", width=10, command=None)
        alpha_label.pack(side=tk.LEFT, padx=1, pady=3)
        
        self.alpha_entry = Entry(frame2p25)
        self.alpha_entry.pack(fill=tk.X, padx=4, expand=False, side=tk.LEFT)
        
        t1_max_label = tk.Label(frame2p25, text="T1 max", width=10)
        t1_max_label.pack(side=tk.LEFT, padx=1, pady=3)
        
        self.t1_max = Entry(frame2p25)
        self.t1_max.pack(fill=tk.X, padx=4, expand=False, side=tk.LEFT)
        
        flair_max_label = tk.Label(frame2p25, text="FLAIR max", width=10)
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
        
        global_ops_label = Label(frame5, text="Global morphological operations", width=25)
        global_ops_label.pack(padx=2, pady=0, side=tk.TOP, anchor=tk.NW)
        
        #finishing_label = Label(frame5, text="Final operations", width=25)
        #finishing_label.pack(padx=2, pady=0, side=tk.TOP, anchor=tk.NE)
        
        self.erode_button = tk.Button(frame5, text="ERODE", width=10, command=lambda: self.alter_all_slices(mor.binary_erosion))
        self.erode_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.dilate_button = tk.Button(frame5, text="DILATE", width=10, command=lambda: self.alter_all_slices(mor.binary_dilation))
        self.dilate_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.open_button = tk.Button(frame5, text="OPEN", width=10, command=lambda: self.alter_all_slices(mor.binary_opening))
        self.open_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.close_button = tk.Button(frame5, text="CLOSE", width=10, command=lambda: self.alter_all_slices(mor.binary_closing))
        self.close_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.zero_button = tk.Button(frame5, text="ZERO", width=10, command=lambda: self.alter_all_slices(self.set_slice_zero))
        self.zero_button.pack(padx=2, pady=1, side=tk.LEFT)

        top_buffer = tk.Label(frame5, text="|", width=5)
        top_buffer.pack(padx=2, pady=0, side=tk.LEFT)        
        
        self.slice_up_button = tk.Button(frame5, text="Slice up", width=12, command=self.slice_up)
        self.slice_up_button.pack(padx=6, pady=1, side=tk.LEFT)

        self.save_mask_button = tk.Button(frame5, text="Save mask", width=12, command=self.write_binarized_file)
        self.save_mask_button.pack(padx=6, pady=1, side=tk.RIGHT)
        
        self.calculate_stats_button = tk.Button(frame5, text="Show statistics", width=12, command=self.stats_popup)
        self.calculate_stats_button.pack(padx=6, pady=1, side=tk.RIGHT)
        

        
        frame6= Frame(self)
        frame6.pack(fill=tk.BOTH, expand=False)
        
        local_ops_label = tk.Label(frame6, text="Local morphological operations         ", width=25)
        local_ops_label.pack(padx=2, pady=0, side=tk.TOP, anchor=tk.NW)
        
        self.local_erode_button = tk.Button(frame6, text="ERODE", width=10, command=lambda: self.alter_current_slice(mor.binary_erosion))
        self.local_erode_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.local_dilate_button = tk.Button(frame6, text="DILATE", width=10, command=lambda: self.alter_current_slice(mor.binary_dilation))
        self.local_dilate_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.local_open_button = tk.Button(frame6, text="OPEN", width=10, command=lambda: self.alter_current_slice(mor.binary_opening))
        self.local_open_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.local_close_button = tk.Button(frame6, text="CLOSE", width=10, command=lambda: self.alter_current_slice(mor.binary_closing))
        self.local_close_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.local_zero_button = tk.Button(frame6, text="ZERO", width=10, command=lambda: self.alter_current_slice(self.set_slice_zero))
        self.local_zero_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        bottom_buffer = tk.Label(frame6, text="|", width=5)
        bottom_buffer.pack(padx=2, pady=0, side=tk.LEFT)
        
        self.slice_down_button = tk.Button(frame6, text="Slice down", width=12, command=self.slice_down)
        self.slice_down_button.pack(padx=6, pady=1, side=tk.LEFT)
        
        self.return_to_binarization_button = tk.Button(frame6, text="Back to binarization", width=12, command=self.return_to_binarization_cmd)
        self.return_to_binarization_button.pack(padx=6, pady=1, side=tk.RIGHT)
        
        self.undo_morph_button = tk.Button(frame6, text="Undo", width=12, command=self.undo_morph)
        self.undo_morph_button.pack(padx=6, pady=1, side=tk.RIGHT)
        
        
        
        frame7= Frame(self)
        frame7.pack(fill=tk.BOTH, expand=False)
        
        roi_label = tk.Label(frame7, text="Region selection operations              ", width=25)
        roi_label.pack(padx=2, pady=0, side=tk.TOP, anchor=tk.NW)
        
        self.lasso_area_button = tk.Button(frame7, text="LASSO", width=10, command=self.lasso_region)
        self.lasso_area_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.finish_lasso_button = tk.Button(frame7, text="CANCEL LASSO", width=10, command=self.finish_lasso)
        self.finish_lasso_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.keep_area_button = tk.Button(frame7, text="Keep", width=10, command=self.keep_lassoed_areas)
        self.keep_area_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        self.exclude_area_button = tk.Button(frame7, text="Delete", width=10, command=self.delete_lassoed_areas)
        self.exclude_area_button.pack(padx=2, pady=1, side=tk.LEFT)
                
        self.fill_area_button = tk.Button(frame7, text="Fill", width=10, command=self.fill_lassoed_areas)
        self.fill_area_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        
        
        frame8= Frame(self)
        frame8.pack(fill=tk.BOTH, expand=False)
        
        roi_label = tk.Label(frame8, text="Geometry sieving operations           ", width=25)
        roi_label.pack(padx=2, pady=0, side=tk.TOP, anchor=tk.NW)
        
        self.gbs_sci_button = tk.Button(frame8, text="GBS SCI", width=10, command=self.gbs_sci)
        self.gbs_sci_button.pack(padx=2, pady=1, side=tk.LEFT)
        
        
        
        self.n_writes = 0
        self.stage = 0


        
        
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
        
        green = np.array([0/255, 255/255, 0/255, 1])
        
        newcolors1[:128, :] = clear_blue
        newcolors1[128:, :] = red
        
        self.probability_cmp = matplotlib.colors.ListedColormap(newcolors1)
        
        newcolors2[:128, :] = clear_navy
        newcolors2[128:, :] = green
        
        self.binary_cmp = matplotlib.colors.ListedColormap(newcolors2)
        
        self.stage = 1
        self.setup_stage(1)
        
        
    def write_binarized_file(self):
        
        print('Writing binarized mask')
        
        the_data = self.current_overlay.copy()
        the_name = os.path.join(self.output_folder, f'binarized_map_v{self.n_writes}.nii.gz')
        companion_name = os.path.join(self.output_folder, f'binarized_map_v{self.n_writes}_stats.csv')
        
        # need to rotate and flip back to original nibabel orientation
        # the_data = np.fliplr(the_data) # uncomment for neurological view, along with the reading function in ugly_helpers
        the_data = np.rot90(the_data, k=3)
        
        out = nib.Nifti1Image(the_data, self.mirage, self.template_header)
        nib.save(out, the_name)
        
        the_dict = self.calculate_stats()
        
        the_series = pd.Series(the_dict)
        the_series.to_csv(companion_name)
        
        self.n_writes += 1
    
    
    def calculate_stats(self):
        
        lesion_volume, n_lesion_voxels, voxel_volume = self.calculate_lesion_vol()
        
        stats_headers = ['lesion_vol', 'n_lesion_voxels', 'infarct_frac', 'flair_vol', 'n_flair_voxels', 'voxel_vol',  'x', 'y', 'z', ]
        stats = [lesion_volume, n_lesion_voxels, lesion_volume/self.brain_vol, self.brain_vol, self.brain_voxels, voxel_volume]
        stats.extend(self.voxel_dims)
        
        the_dict = {h: s for h,s in zip(stats_headers, stats)}
        
        return the_dict
    
    
    def stats_popup(self):
        
        the_dict = self.calculate_stats()
        
        writeout = f'''Lesion volume: {int(the_dict["lesion_vol"])}
        Brain volume: {int(the_dict["flair_vol"])}
        Infarction: {round(the_dict["infarct_frac"]*100,1)}%'''
        
        self.popupmsg(writeout)
        
    
        
    def calculate_lesion_vol(self):
        # returns the lesion vol, n lesion voxels and volume of a voxel as a tuple
        
        n_lesion_voxels = self.current_overlay.sum()
        voxel_volume = np.product(self.voxel_dims)
        lesion_volume = voxel_volume * n_lesion_voxels
        
        return lesion_volume, n_lesion_voxels, voxel_volume
        
    
        
    def return_to_binarization_cmd(self):
        self.current_overlay = self.probability_map
        self.stage = 2
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
            self.trans_button ['state']= tk.NORMAL
            self.trans_entry['state'] = tk.NORMAL
            
            self.radio_t1['state'] = tk.DISABLED
            self.mask_on_checkbox['state'] = tk.DISABLED
            self.radio_flair['state'] = tk.DISABLED
            self.alpha_entry['state'] = tk.DISABLED
            self.t1_max['state'] = tk.DISABLED
            self.flair_max['state'] = tk.DISABLED
            self.binarize_probability_mask_button['state'] = tk.DISABLED
            self.binarize_slider['state'] = tk.DISABLED
            self.slice_slider['state'] = tk.DISABLED
            self.slice_up_button['state'] = tk.DISABLED
            self.slice_down_button['state'] = tk.DISABLED
            
            self.erode_button['state'] = tk.DISABLED
            self.dilate_button['state'] = tk.DISABLED
            self.open_button['state'] = tk.DISABLED
            self.close_button['state'] = tk.DISABLED
            self.zero_button['state'] = tk.DISABLED
            
            self.local_erode_button['state'] = tk.DISABLED
            self.local_dilate_button['state'] = tk.DISABLED
            self.local_open_button['state'] = tk.DISABLED
            self.local_close_button['state'] = tk.DISABLED
            self.local_zero_button['state'] = tk.DISABLED
            
            self.keep_area_button['state'] = tk.DISABLED
            self.exclude_area_button['state'] = tk.DISABLED
            self.fill_area_button['state'] = tk.DISABLED
            self.lasso_area_button['state'] = tk.DISABLED
            self.finish_lasso_button['state'] = tk.DISABLED
            
            self.save_mask_button['state'] = tk.DISABLED
            self.calculate_stats_button['state'] = tk.DISABLED
            self.return_to_binarization_button['state'] = tk.DISABLED
            self.undo_morph_button['state'] = tk.DISABLED
            
            self.gbs_sci_button['state'] = tk.DISABLED
            
        elif stage == 2:
            self.display_label.config(text='Step 2: probability map binarization')
            self.overlay_cmap = self.probability_cmp

            self.bianca_entry['state'] = tk.NORMAL
            self.t1_button['state'] = tk.NORMAL
            self.flair_button ['state']= tk.NORMAL
            self.t1_entry['state'] = tk.NORMAL
            self.flair_entry['state'] = tk.NORMAL
            self.run_bianca_button['state'] = tk.NORMAL
            self.trans_button ['state']= tk.NORMAL
            self.trans_entry['state'] = tk.NORMAL
            
            self.radio_t1['state'] = tk.NORMAL
            self.mask_on_checkbox['state'] = tk.NORMAL
            self.radio_flair['state'] = tk.NORMAL
            self.alpha_entry['state'] = tk.NORMAL
            self.t1_max['state'] = tk.NORMAL
            self.flair_max['state'] = tk.NORMAL
            self.binarize_probability_mask_button['state'] = tk.NORMAL
            self.binarize_slider['state'] = tk.NORMAL
            self.slice_slider['state'] = tk.NORMAL
            self.slice_up_button['state'] = tk.NORMAL
            self.slice_down_button['state'] = tk.NORMAL
            
            self.erode_button['state'] = tk.DISABLED
            self.dilate_button['state'] = tk.DISABLED
            self.open_button['state'] = tk.DISABLED
            self.close_button['state'] = tk.DISABLED
            self.zero_button['state'] = tk.DISABLED
            
            self.local_erode_button['state'] = tk.DISABLED
            self.local_dilate_button['state'] = tk.DISABLED
            self.local_open_button['state'] = tk.DISABLED
            self.local_close_button['state'] = tk.DISABLED
            self.local_zero_button['state'] = tk.DISABLED
            
            self.keep_area_button['state'] = tk.DISABLED
            self.exclude_area_button['state'] = tk.DISABLED
            self.fill_area_button['state'] = tk.DISABLED
            self.lasso_area_button['state'] = tk.DISABLED
            self.finish_lasso_button['state'] = tk.DISABLED
            
            self.save_mask_button['state'] = tk.DISABLED
            self.calculate_stats_button['state'] = tk.DISABLED
            self.return_to_binarization_button['state'] = tk.DISABLED
            self.undo_morph_button['state'] = tk.DISABLED
            
            self.gbs_sci_button['state'] = tk.DISABLED
            
        elif stage == 3:
            self.display_label.config(text='Step 3: binary map adjustment')
            self.overlay_cmap = self.binary_cmp
        
            self.bianca_entry['state'] = tk.NORMAL
            self.t1_button['state'] = tk.NORMAL
            self.flair_button ['state']= tk.NORMAL
            self.t1_entry['state'] = tk.NORMAL
            self.flair_entry['state'] = tk.NORMAL
            self.run_bianca_button['state'] = tk.NORMAL
            self.trans_button ['state']= tk.NORMAL
            self.trans_entry['state'] = tk.NORMAL
            
            self.radio_t1['state'] = tk.NORMAL
            self.mask_on_checkbox['state'] = tk.NORMAL
            self.radio_flair['state'] = tk.NORMAL
            self.alpha_entry['state'] = tk.NORMAL
            self.t1_max['state'] = tk.NORMAL
            self.flair_max['state'] = tk.NORMAL
            self.binarize_probability_mask_button['state'] = tk.DISABLED
            self.binarize_slider['state'] = tk.DISABLED
            self.slice_slider['state'] = tk.NORMAL
            self.slice_up_button['state'] = tk.NORMAL
            self.slice_down_button['state'] = tk.NORMAL
            
            self.erode_button['state'] = tk.NORMAL
            self.dilate_button['state'] = tk.NORMAL
            self.open_button['state'] = tk.NORMAL
            self.close_button['state'] = tk.NORMAL
            self.zero_button['state'] = tk.NORMAL
            
            self.local_erode_button['state'] = tk.NORMAL
            self.local_dilate_button['state'] = tk.NORMAL
            self.local_open_button['state'] = tk.NORMAL
            self.local_close_button['state'] = tk.NORMAL
            self.local_zero_button['state'] = tk.NORMAL
            
            self.keep_area_button['state'] = tk.DISABLED
            self.exclude_area_button['state'] = tk.DISABLED
            self.fill_area_button['state'] = tk.DISABLED
            self.lasso_area_button['state'] = tk.NORMAL
            self.finish_lasso_button['state'] = tk.DISABLED
            
            self.save_mask_button['state'] = tk.NORMAL
            self.calculate_stats_button['state'] = tk.NORMAL
            self.return_to_binarization_button['state'] = tk.NORMAL
            self.undo_morph_button['state'] = tk.NORMAL
            
            self.gbs_sci_button['state'] = tk.NORMAL
            
            
    def slice_up(self):
        
        current = self.slice_slider.get()
        to = current+1
        
        self.slice_slider.set(to)
        
        
    def slice_down(self):
        
        current = self.slice_slider.get()
        to = current-1
        
        self.slice_slider.set(to)

    
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
            #self.setup_stage(1)
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
            
        self.n_writes = 0
                
        self.stage = 2
        self.setup_stage(2)
        
        self.set_output_folder()
        self.flair_file = self.flair_entry.get()
        self.t1_file = self.t1_entry.get()
        
        bianca_master_file = ugh.generate_bianca_master(self.output_folder, self.flair_entry.get(), self.t1_entry.get(), self.trans_entry.get())
        
        self.binarize_slider.set(50)
        
        self.alpha_entry.delete(0, 'end')
        self.alpha_entry.insert(0,0.5)
        
        self.probability_map_file = os.path.join(self.output_folder, 'probability_map.nii.gz')
        ugh.execute_bianca(master=bianca_master_file, model=self.bianca_entry.get(), outname=self.probability_map_file)
        
        self.mask_on_checkbox.select()
        
        self.probability_map = ugh.read_nifti_radiological(self.probability_map_file)
        self.flair = ugh.read_nifti_radiological(self.flair_entry.get())
        self.t1 = ugh.read_nifti_radiological(self.t1_entry.get())
        
        self.sh = self.flair.shape
        self.nx = self.sh[0]
        self.ny = self.sh[1]
        self.nz = self.sh[2]
        
        self.lasso_truth = np.zeros((self.nx, self.ny), bool)
        self.lasso_truth_blank = self.lasso_truth.copy()
        
        self.radio_flair.select()
        
        self.slice_slider.configure(from_=self.flair.shape[2]-1)
        self.slice_slider.set(int(self.flair.shape[2]/2)-1)
        
        self.flair_max.delete(0, 'end')
        self.t1_max.delete(0, 'end')
        
        self.flair_max.insert(0, round(self.flair.max()))
        self.t1_max.insert(0, round(self.t1.max()))
        
        self.current_overlay = self.probability_map

        template = nib.load(self.flair_file)
        self.template_header = template.header
        self.voxel_dims = self.template_header['pixdim'][1:4]
        self.mirage = template.affine
        
        self.voxel_vol = np.product(self.voxel_dims)
        
        brain = template.get_fdata()
        self.brain_voxels = (brain > 0).sum()
        self.brain_vol = self.brain_voxels * self.voxel_vol
        

        
    
    def binarize_probability_mask(self):
        
        self.stage = 3
        self.setup_stage(3)
        
        self.binary_mask = self.probability_map.copy()
        self.binary_mask = self.binary_mask >= self.binarize_slider.get()/100
        self.binary_mask = self.binary_mask.astype(int)
        
        self.current_overlay = self.binary_mask
        
        self.display_scan(self.bg_scan.get(), self.slice_slider.get())
    
    
    def display_scan(self, scan, sli, cmap=matplotlib.cm.gray, end_lasso=True):
  
        
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
        
        if end_lasso:
            self.finish_lasso()
      
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
          
            
    def take_snapshot(self):
        self.snapshot = self.current_overlay.copy()
        
            
    def set_slice_zero(self, sli):
        a = sli.copy()
        a[:] = 0
        return a
            
    
    def binary_operation(self, operation, im2d):
        """
        Applies a binary operation to a 2d image
        

        Parameters
        ----------
        operation : function
            binary operation.
        im2d : 2d np array
            array to apply operation to.

        Returns
        -------
        np array

        """
        
        a = operation(im2d)
        return a
    
    
    def binary_operation_all(self, operation, mat):
        """
        Applies a binary operation to every z slice of a 3d image
        

        Parameters
        ----------
        operation : function
            binary operation.
        mat : 3d np array
            array to apply operation to.

        Returns
        -------
        np array.

        """
        
        n_slices = mat.shape[2]
        a = mat[:]
        for i in range(n_slices):
            a[:,:,i] = self.binary_operation(operation, a[:,:,i])
            
        return a
    
    def undo_morph(self):
        self.current_overlay = self.snapshot.copy()
        self.display_scan(self.bg_scan.get(), self.slice_slider.get())
    
    def alter_current_slice(self, operation):
        self.take_snapshot()
        
        i = self.slice_slider.get()
        self.current_overlay[:,:,i] = self.binary_operation(operation, self.current_overlay[:,:,i])
        self.display_scan(self.bg_scan.get(), self.slice_slider.get())
        
        
    def alter_all_slices(self, operation):
        self.take_snapshot()
        
        self.current_overlay = self.binary_operation_all(operation, self.current_overlay)
        self.display_scan(self.bg_scan.get(), self.slice_slider.get())
        

    def onselect(self, verts):
        p = matplotlib.path.Path(verts)
        current_im = self.current_overlay[:,:,self.slice_slider.get()]
        sh = current_im.shape
        px, py = np.arange(sh[1]), np.arange(sh[1])
        
        xv, yv = np.meshgrid(px, py)
        pix = np.vstack((xv.flatten(), yv.flatten())).T
        
        ind = p.contains_points(pix, radius=0)
        
        truthy = np.reshape(ind, (sh[0], sh[1]))
        
        self.lasso_truth = np.logical_or(self.lasso_truth, truthy)
    
        
    def lasso_region(self):
        
        self.lasso = matplotlib.widgets.LassoSelector(self.plot1, self.onselect)
        
        self.keep_area_button['state'] = tk.NORMAL
        self.exclude_area_button['state'] = tk.NORMAL
        self.fill_area_button['state'] = tk.NORMAL
        self.lasso_area_button['state'] = tk.DISABLED
        self.finish_lasso_button['state'] = tk.NORMAL
        
        
    def finish_lasso(self):
        try:
            del self.lasso
        except AttributeError:
            pass
        try:
            self.lasso_truth = self.lasso_truth_blank
        except AttributeError:
            pass
        
        
        if self.stage == 3:
            self.lasso_area_button['state'] = tk.NORMAL
        
        self.keep_area_button['state'] = tk.DISABLED
        self.exclude_area_button['state'] = tk.DISABLED
        self.fill_area_button['state'] = tk.DISABLED
        self.finish_lasso_button['state'] = tk.DISABLED
        
        self.display_scan(self.bg_scan.get(), self.slice_slider.get(), end_lasso=False)
        
    
    def fill_lassoed_areas(self):
        self.take_snapshot()
        # 1 everything inside the lasso
        self.current_overlay[:,:,self.slice_slider.get()][self.lasso_truth] = 1
        self.display_scan(self.bg_scan.get(), self.slice_slider.get(), end_lasso=True)
    
    
    def keep_lassoed_areas(self):
        self.take_snapshot()
        # zero everything outside the lasso
        self.current_overlay[:,:,self.slice_slider.get()][~self.lasso_truth] = 0
        self.display_scan(self.bg_scan.get(), self.slice_slider.get(), end_lasso=True)        
    
    
    def delete_lassoed_areas(self):
        self.take_snapshot()
        # 0 everything inside the lasso
        self.current_overlay[:,:,self.slice_slider.get()][self.lasso_truth] = 0
        self.display_scan(self.bg_scan.get(), self.slice_slider.get(), end_lasso=True)
    

    
    def popupmsg(self, msg, title='!!! ATTENTION !!!'):
        popup = tk.Tk()
        popup.wm_title(title)
        label = tk.Label(popup, text=msg)
        label.pack(side="top", fill="x", pady=10, padx=10)
        B1 = tk.Button(popup, text="Got it", command = popup.destroy)
        B1.pack()
        popup.mainloop()
        
        
    def gbs_sci(self):
        self.take_snapshot()
        
        self.current_overlay = gbs.sieve_image(self.current_overlay)
        
        self.display_scan(self.bg_scan.get(), self.slice_slider.get(), end_lasso=True)
        

def main():

    root = Tk()
    root.geometry("1300x1100+400+200")
    app = MainApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()