# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914
"""
Process raw dataset for experiments
"""

import sys
sys.path.append("../../src")
sys.path.append("../../")
import argparse
from importlib import reload
import logging
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from notebooks.utils import global_utils
import numpy as np
from glob import glob

def load_results():
    results_dict = defaultdict(list)
    folder_names = ['dbscan','kmeans']
    subfolder_names = {'stock':'Stock','intel':'Intel oneAPI Scikit-Learn Extension'}

    for experiment_n, foldername in enumerate(folder_names):
        for subfolder_name in subfolder_names.keys():
            for log_n, logname in enumerate(glob(f"../logs/{foldername}/{subfolder_name}_large_*.log")):
                logfile = logname
                with open(logfile, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    if line.find('Number of data points') != -1:
                        number_points = int(re.findall("\d+",line)[0])
                        str_number_points = global_utils.milify(number_points)
                        results_dict['number of pointss'].append(str_number_points)
                        results_dict['number of points'].append(number_points)
                        break               
                filtered_lines = [line for line in lines if line.find('Total hyperparameter tuning time') != -1]
                time = np.mean([float(re.findall("\d+.\d+",i)[0]) for i in filtered_lines])
                results_dict['method'].append(foldername)
                results_dict['type'].append(subfolder_name)
                results_dict['time'].append(time)
    return results_dict

def print_kmeans_plot():
    df = pd.DataFrame(load_results()).sort_values(by=['method','type','number of points'])
    fig, ax1 = plt.subplots(1,figsize=[14,6])
    fig.suptitle('Performance Speedup Relative to Stock Scikit-Learn \n (DBScan Hyperparameter Analysis-21 features)')
    ax1.set_ylabel('Relative Performance to Stock \n (Higher is better)')
    method = 'dbscan'
    sub_df = df[df['method']==method]
    stock = sub_df['time'][sub_df['type']=='stock']
    intel = sub_df['time'][sub_df['type']=='intel']
            
    global_utils.bar_comparison(stock,intel,intel_label='Intel oneAPI Scikit-Learn Extension',ax=ax1,xlabel=f'Number of Samples',
                xticks=sub_df['number of pointss'][sub_df['type']=='intel'].unique(),legend=True, relative=True)

def print_dbscan_plot():
    df = pd.DataFrame(load_results()).sort_values(by=['method','type','number of points'])
    fig, ax1 = plt.subplots(1,figsize=[14,6])
    fig.suptitle('Performance Speedup Relative to Stock Scikit-Learn \n (KMeans Hyperparameter Analysis-21 features)')
    width = 0.35 
    ax1.set_ylabel('Relative Performance to Stock \n (Higher is better)')
    method = 'kmeans'
    sub_df = df[df['method']==method]
    stock = sub_df['time'][sub_df['type']=='stock']
    intel = sub_df['time'][sub_df['type']=='intel']

    global_utils.bar_comparison(stock,intel,intel_label='Intel oneAPI Scikit-Learn Extension',ax=ax1,xlabel=f'Number of Samples',
                xticks=sub_df['number of pointss'][sub_df['type']=='intel'].unique(),legend=True, relative=True)