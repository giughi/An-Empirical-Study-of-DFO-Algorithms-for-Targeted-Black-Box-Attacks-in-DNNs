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
parser.add_argument("--Data", default='Imagenet', help="Dataset that we want to attack. At the moment we have:'MNIST','CIFAR','STL10','Imagenet','MiniImageNet'")
parser.add_argument("--title", default='', help="This will be associated to the image title")
parser.add_argument("--plot_type", default='CDF', help="What graph we generate; `CDF` or `quantile`")
parser.add_argument("--save",default=True, help="Boolean to save the result", type=bool)
parser.add_argument('--Adversary', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--zoom', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--both', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--legend', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--subspace_attack', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument("--loc", default=1, help="Subdomain Dimension", type=int)

# parser.add_argument("--Adversary", default=False, action='store_true', help="Boolean for plotting adversarial attacks too")
parser.add_argument("--p_init", default=0.01, help="If in `quantile` option, it says which quantile to plot", type=float)
parser.add_argument("--max_iter", default=15000, help="Maximum iterations allowed", type=int)
parser.add_argument("--sub_dim", default=1000, help="Subdomain Dimension", type=int)
parser.add_argument("--second_iter",default=False, help="Loading results from second round of attacks", type=bool)
parser.add_argument("--only_second",default=False, help="Loading results from second round of attacks", type=bool)
parser.add_argument("--adv_inception",default=False, help="Using adversary results", type=bool)
args = parser.parse_args()

# just to load the right file
if not args.subspace_attack:
    args.sub_dim=None

# If we want to plot both the adversary and not results on teh same graph
if args.both:
    args.Adversary = False

if args.Data=='Imagenet':
    max_queries = 15000
else:
    max_queries = 3000
args.max_iter = max_queries

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

list_possible_attacks = ['boby', 'combi', 'square', 'gene', 'FW']

list_available_results = []
list_available_results_adv = []
list_available_attacks = []

# Let's import the different results that we can achieve
for attack in list_possible_attacks:
    try:
        results = import_results(attack, args)
    except:
        results = None

    if results is not None:
        list_available_results.append(results)
        list_available_attacks.append(attack)

if args.both:
    for attack in list_possible_attacks:
        results_adv = import_results(attack, args, both=True)
        list_available_results_adv.append(results_adv)


name_possible_attacks = map_to_complete_names(list_available_attacks, args.both)

# Let's define the saving name of the plot
saving_title=(main_dir+'/Results/'+str(args.Data)+'/Plots/CDF_adversary_'+str(args.Adversary) 
            + args.title + '_sub_dim_' + str(args.sub_dim) + '_eps_' + str(args.eps) + '_both_' 
            + str(args.both) + '_legend_' + str(args.legend) +'_zoom_' +str(args.zoom) + '.pdf')

# Let's print the plots of the results
generating_cumulative_blocks(list_available_results, 
                            list_available_results_adv, 
                            name_possible_attacks,
                            max_queries,
                            1000,  #interpolation points
                            args.eps,
                            saving_title, 
                            legend=args.legend, 
                            zoom=args.zoom,
                            both=args.both,
                            loc=args.loc);

# Let's print the information the data used
print_table_data(list_available_results, 
                list_available_results_adv, 
                name_possible_attacks,
                max_queries,)
