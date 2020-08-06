import os
import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np
import random
import time

import pickle

from PIL import Image


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p

def square_attack_linf(model, x, y, eps, n_iters, p_init, targeted, loss_type, print_every=50,
                       subspace_attack=False, subspace_dim=0):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    min_val, max_val = -0.5, 0.5 if x.max() <= 1 else 255
    h, w, c = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    
    # for the subspace attack we have to identify the positions
    if subspace_attack:
        x_position, y_position, c_position = cohordinates_most_variance(x, subspace_dim)
        print(len(x_position))

    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], 1, w, c])
    x_best = np.clip(x, min_val, max_val)

    logits = model.predict(x_best)
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    print('Initial Loss = ', loss_min)
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        x_curr, x_best_curr, y_curr = x, x_best, y
        loss_min_curr, margin_min_curr = loss_min, margin_min
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
                        
            if not subspace_attack:
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)
                x_curr_window = x_curr[i_img, center_h:center_h+s, center_w:center_w+s, :]
                x_best_curr_window = x_best_curr[i_img, center_h:center_h+s, center_w:center_w+s, :]

                while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, center_h:center_h+s, center_w:center_w+s, :], 
                                            min_val, max_val) 
                                    - x_best_curr_window) 
                            < 10**-7) == c*s*s:
                    deltas[i_img, center_h:center_h+s, center_w:center_w+s, :] = np.random.choice([-eps, eps], size=[1, 1, c])
            else:
                s = 1
                elem = np.random.randint(0, subspace_dim)
                center_h = y_position[elem]
                center_w = x_position[elem]
                center_c = c_position[elem]

                x_curr_window = x_curr[i_img, center_h:center_h+s, center_w:center_w+s, center_c]
                x_best_curr_window = x_best_curr[i_img, center_h:center_h+s, center_w:center_w+s, center_c]
                
                while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, center_h:center_h+s, center_w:center_w+s, center_c], 
                                            min_val, max_val) 
                                    - x_best_curr_window) 
                            < 10**-7) == 1*s*s:
                    deltas[i_img, center_h:center_h+s, center_w:center_w+s, center_c] = np.random.choice([-eps, eps], size=[1, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        logits = model.predict(x_new)
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min = (idx_improved) * loss + (~idx_improved) * loss_min_curr
        margin_min = (idx_improved) * margin + (~idx_improved) * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best = (idx_improved) * x_new + (~idx_improved) * x_best_curr
        n_queries += 1
        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start

        if np.mod(i_iter, print_every)==0:
            print('[L1] {}: margin={}, loss ={:.3f}, p={}  (n_ex={}, eps={:.3f}, {:.2f}s)'.
                format(i_iter+1, margin_min[0], loss_min[0], p, x.shape[0], eps, time_total))

        if acc == 0:
            break
    
    if len(x_best.shape) == 3:
            x_best = x_best.reshape((1,) + x_best.shape)
    return x_best , n_queries, [], acc==0



#########################################################
# Functions related to the optimal sampling of an image #
#########################################################


def find_neighbours(r, c, k, n, m, R):
    # This computes the neihgbours of a pixels (r,c,k) in an image R^(n,m,R)
    # Note: We never consider differnt layers of the RGB
    neighbours = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if ((r+i) >= 0) and ((r+i) < n):
                if ((c+j) >= 0) and ((c+j) < m):
                    if not((i, j) == (0, 0)):
                        neighbours.append([0, r+i, c+j, k])
    return neighbours


def get_variation(img, neighbours):
    list_val = []
    for i in range(len(neighbours)):
        list_val.append(img[neighbours[i][1]][neighbours[i][2]][neighbours[i][3]])
    sum_var = np.std(list_val)
    return sum_var

    
def total_img_var(row, col, k, img):
    n, m, RGB = img.shape
    neighbours = find_neighbours(row, col, k, n, m, RGB)
    total_var = get_variation(img, neighbours)
    return total_var
    
    
def image_region_importance(img,k):
    """
    This function obtaines the image as an imput and it computes the importance of the 
    different regions. 
    
    Inputs:
    - img: tensor of the image that we are considering
    - k: number of channels in the perturbation that we are considering
    Outputs:
    - probablity_matrix: matrix with the ecorresponding probabilities.
    """
    n, m, _ = img.shape
    probability_matrix = np.zeros((n, m, k))
    for i in range(k):
        for row in range(n):
            for col in range(m):
                probability_matrix[row, col, i] = total_img_var(row, col, i, img)
    # We have to give a probability also to all the element that have zero variance
    # this implies that we will add to the whole matrix the minimum nonzero value, divided
    # by 100
    probability_matrix += np.min(probability_matrix[np.nonzero(probability_matrix)])/100
    # Normalise the probability matrix
    probability_matrix = probability_matrix/np.sum(probability_matrix)
    return probability_matrix

#########################################################
# Functions to subdivide into sub regions the full pixels#
#########################################################

def associate_block(A, i, j, k, nn_i, nn_j, association):
    for ii in range(int(i), int(i + nn_i)):
        for jj in range(int(j), int(j + nn_j)):
            A[0, ii, jj, k] = int(association)
    return A


def matr_subregions_division(img_size, num_channels, n, channel_size):
    """
    This allows to compute the matrix fo dimension equal to the image with the
    region of beloging for each supervariable with only a block composition

    :param img_size: Dimension of the image, i.e. m.
    :param n: Dimension of the super grid that we are using (n,n,c).
    :param channel_size: number of channels c in the pertubation.

    :return: The matrix with the supervariable index to which each pixel belongs
    """
    A = np.zeros((1,img_size,img_size,num_channels))
    partition = []
    nn_up = np.ceil(img_size/n)
    for i in range(n):
        partition.append(int(i*nn_up))
    partition.append(img_size)
    # check that the number of intervals is n
    if len(partition)!=n+1:
        print('[WARNING] The partition is not exact.')
    association = 0
    for k in range(3):
        for i in range(n):
            xi = partition[i]
            di = partition[i+1]-partition[i]
            for j in range(n):
                xj = partition[j]
                dj = partition[j+1]-partition[j]
                A = associate_block(A, xi, xj, k, di, dj, association)
                association += 1
        # If we just have one channel, we can associate to all the channels the same
        # pattern.
        if channel_size==1:
            association=0
    return A


def cohordinates_most_variance(img, subspace_dim):
    # Find dimension of the image
    h, w, c = img.shape[1:]
    # Generate a tensor with all the numeration and their importance
    super_dependency = matr_subregions_division(h, c, h, c)
    prob = image_region_importance(img[0], c).reshape(-1,)
    # select most important indices of the image
    top_n_indices = np.argsort(prob)[-subspace_dim:]
    # find the x and y and c cohordinates of the image
    x_position = []
    y_position = []
    c_position = []
    for i in top_n_indices:
        # print(i)
        c_position.append(int(np.floor(i/(h*w))))
        y_position.append(int(np.floor((i - c_position[-1]*h*w)/h)))
        x_position.append(int(i - c_position[-1]*h*w - y_position[-1]*h))

    # checking that the mapping is working
    # A = 0 *img.copy()
    # B = 0 *img.copy().reshape(-1)
    # A[0, y_position[1], x_position[1],c_position[1]] = 1e5
    # B[super_dependency.reshape(-1) == top_n_indices[1]] = 1e5
    # B = B.reshape(img.shape)
    # print('THE difference is', np.linalg.norm((B-A)), np.amax(B-A))
    # print('numpy says', np.argwhere(super_dependency==top_n_indices[1]))
    # print('we suggest', y_position[1],x_position[1],c_position[1])

    return x_position, y_position, c_position

    


