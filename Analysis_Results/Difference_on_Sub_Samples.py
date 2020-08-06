import os
import sys
import numpy as np
import random
import time

from utils_CDF_plotting import *
from utils_uploading import *

import pickle

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
from itertools import chain

from mpl_toolkits.axes_grid.inset_locator import inset_axes

import argparse

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')



parser = argparse.ArgumentParser()
parser.add_argument("--eps", default=0.1, help="Energy of the pertubation", type=float)
# parser.add_argument("--eps", default=[0.1], nargs='+', type=float, help="Energy of the pertubation")
parser.add_argument("--Data", default='MNIST', help="Dataset that we want to attack. At the moment we have:'MNIST','CIFAR'")
parser.add_argument("--title", default='', help="This will be associated to the image title")
parser.add_argument("--plot_type", default='CDF', help="What graph we generate; `CDF` or `quantile`")
parser.add_argument("--save",default=True, help="Boolean to save the result", type=bool)
parser.add_argument('--legend', type=str_to_bool, nargs='?', const=True, default=False)

parser.add_argument("--max_iter", default=15000, help="Maximum iterations allowed", type=int)
args = parser.parse_args()

# just to load the right file
max_queries = 3000
args.max_iter = max_queries

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

list_possible_attacks = ['dist', 'rand', 'orde']

list_available_results = []
list_available_results_adv = []
list_available_attacks = []

# Let's import the different results that we can achieve
for attack in list_possible_attacks:
    try:
        results = import_results_sub(attack, args)
    except:
        results = None

    if results is not None:
        list_available_results.append(results)
        list_available_attacks.append(attack)


name_possible_attacks = map_to_complete_names_sub(list_available_attacks)

# Let's define the saving name of the plot
saving_title=(main_dir+'/Results/'+str(args.Data)+'/Plots/Subsampling_CDF_adversary_'
            + args.title + '_eps_' + str(args.eps) + '_legend_' + str(args.legend) +'.pdf')

# Let's print the plots of the results
generating_cumulative_blocks(list_available_results, 
                            list_available_results_adv, 
                            name_possible_attacks,
                            max_queries,
                            1000,  #interpolation points
                            args.eps,
                            saving_title, 
                            legend=args.legend, 
                            zoom=False,
                            both=False,
                            loc=0);
