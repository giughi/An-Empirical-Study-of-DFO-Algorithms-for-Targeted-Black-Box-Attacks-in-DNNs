
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from itertools import chain

from mpl_toolkits.axes_grid.inset_locator import inset_axes
from utils_uploading import *

def computation_SR(results, max_queries):
    """
    This function compyutes the SR of a list of results that it receives as an input
    """

    n_attacks = len(results)
    SR_list = []
    # find minimal number of attacks
    m = np.Inf
    for j in range(n_attacks):
        m = int(np.minimum(m,len(results[j])))

    # compute the SR
    for i in range(n_attacks):
        result = results[i]
        count = 0
        for j in range(m):
            if result[j]<max_queries:
                count +=1
        SR_list.append(count/m)

    return SR_list

def generating_SR_plot(epsilons, SR_list, epsilons_adv, SR_list_adv, name_arrays,args, title):    
    """
    Function to plot the list_arrays of results relative to the name_arrays
    """
    # Let's first find the minimum number of attacks per class
    n = len(SR_list[0])
    if n!=len(name_arrays):
        print('[ERROR] The names and list are not the same')
        return(-1)
    
    m = len(epsilons)
        
    M = np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):
            M[i,j] = SR_list[j][i]

    # If we have both, will save the CDFs now
    if args.both:
        n2 = len(SR_list_adv[0])
        if n2!=len(name_arrays):
            print('[ERROR] The names and list are not the same')
            return(-1)
        
        m2 = len(epsilons_adv)
            
        M2 = np.zeros((n2,m2))
        
        for i in range(n2):
            for j in range(m2):
                M2[i,j] = SR_list_adv[j][i]

    # Let's plot the main results in list_arrays
    fontSize = 18

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig,ax = plt.subplots()
    plt.grid()
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    
    normal_ones = []
    adv_ones = []
    for i in range(n):
        non_zero_index = find_non_zero_index(M[i,:])
        eps_array = np.array(epsilons)
        # print(eps_array[[0,1]])#[0,non_zero_index])
        # print(M[i,non_zero_index])
        ax.loglog(eps_array[non_zero_index],M[i,non_zero_index],'o-',color=colors[i],lw=2)
        normal_ones.append(ax.lines[-1])
        if args.both:
            for j in range(n2):
                non_zero_index = find_non_zero_index(M2[j,:])
                eps_array_adv = np.array(epsilons_adv)
                ax.loglog(eps_array_adv[non_zero_index],M2[j,non_zero_index],'x--',color=colors[j],lw=2)
                adv_ones.append(ax.lines[-1])
    

    if args.legend:
        # if not args.both:
        #     plt.legend(name_arrays,loc=loc, fontsize=fontSize, framealpha=0.4)
        # else:
        l1 = plt.legend(normal_ones,['','','','',''],loc=1, fontsize=16, framealpha=0.1,ncol=1,
            bbox_to_anchor=(1.25,0.5))
        l2 = plt.legend(adv_ones,name_arrays,loc=1, fontsize=16, framealpha=0.1,ncol=1, 
            bbox_to_anchor=(1.7,0.5))

        plt.gca().add_artist(l1)
        if args.Data=='CIFAR':
            plt.text(0.205,0.045,'Non-Adv', fontsize=14)
            plt.text(0.42,0.045,'Adv', fontsize=14)
        else:
            plt.text(0.125,0.05,'Non-Adv', fontsize=14)
            plt.text(0.205,0.05,'Adv', fontsize=14)

    plt.xlabel(r"$\epsilon_\infty$",fontsize=18)
    plt.ylabel('SR',fontsize=fontSize)
    # plt.axis([0, max_eval, 0 ,1],fontsize=fontSize)
    
    fig.savefig(title,bbox_inches='tight')
        

    return M