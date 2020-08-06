import os
import sys
import numpy as np
import random
import time

import pickle

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)
from itertools import chain


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save",default=True, help="Boolean to save the result", type=bool)
parser.add_argument("--max_iter", default=8000, help="Maximum iterations allowed", type=int)
args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

L_inf_var = 0.1
saving = args.save
maxiter = args.max_iter

loss_y = []
loss_n = []

    
with open(main_dir+'/Results/Imagenet/Hierarchical_Effect/Iterative_0.1_batch_dim_50_max_queries_8000_hier_True.txt', "rb") as fp:
    L_y = pickle.load(fp)
with open(main_dir+'/Results/Imagenet/Hierarchical_Effect/Iterative_0.1_batch_dim_50_max_queries_8000_hier_False.txt', "rb") as fp:
    L_n = pickle.load(fp)

# unwrapping the summaries

print(len(L_y))

# summary_y = []
len_y = len(L_y[0])

for i in range(len_y):
    for iteration in range(len(L_y[0][i])):
        temp = L_y[0][i][iteration]['fvals'].values
        for j in range(len(temp)):
            loss_y.append(temp[j][0])

x_y = np.arange(len(loss_y))

len_n = len(L_n[0])
for i in range(len_n):
    for iteration in range(len(L_n[0][i])):
        temp = L_n[0][i][iteration]['fvals'].values
        for j in range(len(temp)):
            loss_n.append(temp[j][0])

x_n = np.arange(len(loss_n))


saving_title=main_dir+'/Results/Imagenet/Plots/Lifting_effect.pdf'

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rc('text',usetex = True)
                
fig  = plt.figure()

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

# plt.plot(r,np.transpose(M[:3]),lw=2.5)
xchanges = [56,280,1176,4648]
plt.plot(x_y,loss_y,lw=2.5,linestyle='-')
plt.plot(x_n,loss_n/loss_n[0]*loss_y[0],lw=2.5,linestyle='-')

plt.legend(['With Lifting','No Lifting'], fontsize=18, framealpha=0.4, loc=1)
for xchange in xchanges:
    plt.axvline(x=xchange, color='g', lw=1.5)
plt.xlabel("Queries",fontsize=18)
plt.ylabel(r"$\mathcal{L}$",fontsize=18)
plt.axis([0,8000,1.5,loss_y[0]],fontsize=18)

fig.savefig(saving_title,bbox_inches='tight')