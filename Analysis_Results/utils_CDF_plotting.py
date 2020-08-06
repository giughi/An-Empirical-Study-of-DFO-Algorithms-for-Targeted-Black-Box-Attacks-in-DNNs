
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from itertools import chain

from mpl_toolkits.axes_grid.inset_locator import inset_axes


def generating_cumulative_blocks(list_arrays, list_arrays_both, name_arrays, 
                                 max_eval, refinement,BATCH,title,legend,
                                 zoom,both,loc):    
    """
    Function to plot the list_arrays of results relative to the name_arrays
    """
    # Let's first find the minimum number of attacks per class
    n = len(list_arrays)
    if n!=len(name_arrays):
        print('[ERROR] The names and list are not the same')
        return(-1)
    m = np.Inf
    for j in range(n):
        m = int(np.minimum(m,len(list_arrays[j])))
    # Let's generate a matrix with the values of the CDF that we will pirnt
    r = np.array(range(refinement+1))*max_eval/refinement
    
    M = np.zeros((n,len(r)))
    
    for i in range(n):
        results = list_arrays[i][:m]
        for j in range(len(r)):
            M[i,j] = np.sum(results<r[j])/len(results)

    # If we have both, will save the CDFs now
    if both:
        if zoom:
            print('[ERROR] Both zoom and both active flags')
        n2 = len(list_arrays_both)
        if n2!=len(name_arrays):
            print('[ERROR] The names and list are not the same')
            return(-1)
        m2 = np.Inf
        for j in range(n2):
            m2 = int(np.minimum(m2,len(list_arrays_both[j])))
            
        # Let's generate a matrix with the values of the CDF that we will pirnt
        r2 = np.array(range(refinement+1))*max_eval/refinement
        
        print('Size',n2,len(r2), n, len(r))
        M2 = np.zeros((n2,len(r2)))
        
        for i in range(n2):
            results = list_arrays_both[i][:m2]
            for j in range(len(r2)):
                M2[i,j] = np.sum(results<r2[j])/len(results)

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
        ax.plot(r,M[i,:],color=colors[i],lw=2)
        normal_ones.append(ax.lines[-1])
        if both:
            for i in range(n2):
                ax.plot(r2,M2[i,:],'--',color=colors[i],lw=2)
                adv_ones.append(ax.lines[-1])
    

    if legend:
        if not both:
            plt.legend(name_arrays,loc=loc, fontsize=fontSize, framealpha=0.4)
        else:
            l1 = plt.legend(normal_ones,['','','','',''],loc=1, fontsize=16, framealpha=0.1,ncol=1,
                bbox_to_anchor=(0.55,0.5))
            l2 = plt.legend(adv_ones,name_arrays,loc=1, fontsize=16, framealpha=0.1,ncol=1, 
                bbox_to_anchor=(1,0.5))

            plt.gca().add_artist(l1)
            plt.text(1050/3000*max_eval,0.48,'Non-Adv', fontsize=14)
            plt.text(1600/3000*max_eval,0.48,'Adv', fontsize=14)

    plt.xlabel('Queries',fontsize=fontSize)
    plt.ylabel('CDF',fontsize=fontSize)
    plt.axis([0, max_eval, 0 ,1],fontsize=fontSize)
    
    if zoom:
        a = plt.axes([.45, .45, .4, .4],)
        dimni = 5
        plt.grid()
        for i in range(n):
            plt.plot(r,M[i,:],color=colors[i],lw=2)
        plt.xlabel('Queries',fontsize=fontSize-dimni)
        plt.ylabel('CDF',fontsize=fontSize-dimni)
        plt.axis([max_eval*0.25, max_eval, 0 ,0.21],fontsize=fontSize-dimni)
        plt.xticks(fontsize=fontSize-dimni)
        plt.yticks(fontsize=fontSize-dimni)

    
    
    fig.savefig(title,bbox_inches='tight')
        

    return M

def print_table_data(list_normal, list_adv, names, max_queries):
    """
    Function to print the numerical results for each attack, i.e. SR, median and mean queries 
    in succesfull attacks
    """
    # let's consider each case seaparetely
    for i in range(len(names)):
        attack = names[i]
        #let's first consider the normally trained cases
        successful_attacks = []
        for j in range(len(list_normal[i])):
            if list_normal[i][j] < max_queries:
                successful_attacks.append(list_normal[i][j])
        
        # and now the adversarial case
        successful_attacks_adv = []
        for j in range(len(list_adv[i])):
            if list_adv[i][j] < max_queries:
                successful_attacks_adv.append(list_adv[i][j])
            
        print("===Attack: ", attack)
        print("*norm* Total Number:", len(list_normal[i]), ", SR:", len(successful_attacks)/len(list_normal[i]), 
              ", median:", np.median(successful_attacks), ", mean: ", np.mean(successful_attacks))
        print("*adve* Total Number:", len(list_adv[i]), ", SR:", len(successful_attacks_adv)/len(list_adv[i]), 
              ", median:", np.median(successful_attacks_adv), ", mean: ", np.mean(successful_attacks_adv))

def print_data_at_query(list_normal, names, query):
    """
    Function to print the numerical results for each attack, at a specific query
    """
    # let's consider each case seaparetely
    for i in range(len(names)):
        attack = names[i]
        #let's first consider the normally trained cases
        successful_attacks = []
        for j in range(len(list_normal[i])):
            if list_normal[i][j] < query:
                successful_attacks.append(list_normal[i][j])
            
        print("===Attack: ", attack)
        print("*norm* Total Number:", len(list_normal[i]), ", CDF:", len(successful_attacks)/len(list_normal[i]))