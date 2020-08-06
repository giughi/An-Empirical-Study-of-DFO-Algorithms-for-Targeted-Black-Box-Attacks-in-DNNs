import os
import sys
import numpy as np
import random
import time

from utils_CDF_plotting import *
from utils_uploading import *
from utils_SR_plotting import *

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
parser.add_argument('--both', type=str_to_bool, nargs='?', const=True, default=True)
parser.add_argument('--legend', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--subspace_attack', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument("--loc", default=1, help="Subdomain Dimension", type=int)

# parser.add_argument("--Adversary", default=False, action='store_true', help="Boolean for plotting adversarial attacks too")
parser.add_argument("--p_init", default=0.01, help="If in `quantile` option, it says which quantile to plot", type=float)
parser.add_argument("--quantile", default=0.5, help="If in `quantile` option, it says which quantile to plot", type=float)
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
    args.max_iter=3000

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

list_possible_attacks = ['boby', 'combi', 'square', 'gene', 'FW']

list_available_results = []
list_available_results_adv = []
list_available_attacks = []

#Let's define the epsilons on which we compute the SR
if args.Data=='CIFAR':
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15]
    epsilons_adv = [0.02, 0.05, 0.1, 0.15]
elif args.Data=='Imagenet':
    epsilons = [0.01, 0.02, 0.05, 0.1]
    epsilons_adv = [0.02, 0.05, 0.1]



# Let's import the different results that we can achieve
args.Adversary = False

SR_list = []
for eps in epsilons:
    args.eps = eps
    results_list = []
    
    for attack in list_possible_attacks:
        results_list.append(import_results(attack, args))
    
    SR_list.append(computation_SR(results_list, max_queries))


args.Adversary = True
SR_list_adv = []
for eps in epsilons_adv:
    args.eps = eps
    results_list = []
    
    for attack in list_possible_attacks:
        results_list.append(import_results(attack, args))
    
    SR_list_adv.append(computation_SR(results_list, max_queries))


# Identify the name of the attacks
name_possible_attacks = map_to_complete_names(list_possible_attacks, args.both)


saving_title=(main_dir+'/Results/'+str(args.Data)+'/Plots/SR_both_' 
            + str(args.both) + '_legend_' + str(args.legend) + '.pdf')
# Plot the results
generating_SR_plot(epsilons, SR_list, epsilons_adv, SR_list_adv, name_possible_attacks,args, 
                  saving_title)