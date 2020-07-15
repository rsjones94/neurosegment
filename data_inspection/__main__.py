#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for data_inspection. This script reads in
tabular patient data and analyzes it for outliers. First, it inspects specified
columns for data integrity (missing values) and produces histograms if appropriate.

Then it analyzes specified 2d relationships, producing scatter plots and identifying
outliers.

Finally it runs the DBSCAN algorithm to flag any potential outliers.

Note that on my machine this uses the venv "tabular_analysis"
"""

import os
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl

from support import is_empty, numbery_string_to_number


data_path = r'/Users/manusdonahue/Documents/Sky/SCD_pt_data_labels_piped.csv'
out_folder = r'/Users/manusdonahue/Documents/Sky/data_inspection/testing' # should not exist

# column that contains the unique deidentified patient ID
study_id_col = 'Study ID'

# columns we want to inspect for completeness and produce histograms/barplots for
# each key is a column name, and the value is True if there MUST be a value and
# False if there does not need to be a value. If there must be a value if and
# only if another column(s) is filled, then the value should be a list of those columns
single_cols = {
               'Age': True,
               'Race': True,
               'Hemoglobin genotype': True,
               'Gender': True,
               'BMI': True,
               'Specify total HU daily dosage (mg)': ['Hydroxyurea'],
               'HTN': True,
               'Diabetes': True,
               'Coronary artery disease': True,
               'High cholesterol': True,
               'Hgb': True,
               'Hct/PCV': True
               }

# 2d relationships we want to use to check for outliers. [independent, dependent]
# numeric data only pls
double_cols = [['Specify total HU daily dosage (mg)', 'MCV']]


#######################################

###### setup

mono_folder = os.path.join(out_folder, 'mono')
bi_folder = os.path.join(out_folder, 'bi')
multi_folder = os.path.join(out_folder, 'multi')

overview_report = os.path.join(out_folder, 'overview.txt')
missing_data_report = os.path.join(out_folder, 'missing_data.csv')
outliers_report = os.path.join(out_folder, 'outliers.csv')

try:
    os.mkdir(out_folder)
except FileExistsError:
    no_answer = True
    while no_answer:
        ans = input('The output directory exists. Overwrite? [y/n]\n')
        if ans == 'y':
            no_answer = False
            shutil.rmtree(out_folder)
            os.mkdir(out_folder)
        elif ans == 'n':
            raise FileExistsError('File exists. Process aborted')
        else:
            print('Response must be "y" or "n"')
    
os.mkdir(mono_folder)
os.mkdir(bi_folder)
os.mkdir(multi_folder)

df = pd.read_csv(data_path, sep='|', low_memory=False, dtype={study_id_col:'object'})

problem_pts_cols = [study_id_col]
problem_pts_cols.extend(single_cols.keys())
problem_pts = pd.DataFrame(columns=problem_pts_cols)
problem_pts = problem_pts.set_index('Study ID') # this data will relate pt IDs to a list of columns for which data
# is missing, iff that missing data is marked as essential (by the variable single_cols)

outlier_pts = {} # this data will relate pt IDs to a list of columns for which
# the data seems to be an outlier

###### plot and inspect the monodimensional data
problem_patients_dict = {}
for col in single_cols:
    data = df[col]
    plt.figure()
    plt.title(col)
    plt.ylabel('Count')
    
    print(f'Plotting: {col}. dtype is {data.dtype}')
    if data.dtype == 'object':
        counts = Counter(data)
        if np.nan in counts:
            counts['nan'] = counts[np.nan]
            del counts[np.nan]
        names = list(counts.keys())
        values = list(counts.values())
        plt.bar(names, values)
    else:
        plt.hist(data)
        plt.xlabel('Value')
    
    scrub_col = col.replace('/', '-') # replace slashes with dashes to protect filepath
    fig_name = os.path.join(mono_folder, f'{scrub_col}.png')
    plt.savefig(fig_name)
    plt.close()

    print(f'Evaluating completeness')
    for i, row in df.iterrows():
        # explicit comparisons of bools needed because we are exploiting the ability to mix key datatypes
        if not is_empty(row[col]) or single_cols[col] is True: # if the entry isn't nan or if data isn't needed
            has_data = True
        elif single_cols[col] is False: # if data is required
            has_data = False
        else: # if we get here, need to see if the companion columns are filled
            # if all companion columns are filled, then data is required
            print(f'Checking to see companions are filled')
            companions = [row[c] for c in single_cols[col]]
            has_required_companions = all([not is_empty(row[c]) for c in single_cols[col]])
            has_data = not has_required_companions
            
        if not has_data:
            pt_id = row[study_id_col]
            try:
                problem_patients_dict[pt_id].append(col)
            except KeyError:
                problem_patients_dict[pt_id] = [col]
            
            
# write the missing data report
for pt, cols in problem_patients_dict.items():
    insert = pd.Series({col:1 for col in cols}, name=pt)
    problem_pts = problem_pts.append(insert, ignore_index=False)
problem_pts = problem_pts.sort_index()
problem_pts.to_csv(missing_data_report)

###### do the 2d analyses
for ind_col, dep_col in double_cols:
    fig_name = os.path.join(bi_folder, f'{dep_col}-v-{ind_col}.png')
    plt.figure()
    plt.title(f'{dep_col} vs. {ind_col}')
    
    x = df[ind_col]
    y = df[dep_col]
    
    x = [numbery_string_to_number(i) for i in x]
    y = [numbery_string_to_number(i) for i in y]
    
    
    plt.scatter(x, y)
    plt.xlabel(ind_col)
    plt.ylabel(dep_col)
    plt.savefig(fig_name)
    plt.close()