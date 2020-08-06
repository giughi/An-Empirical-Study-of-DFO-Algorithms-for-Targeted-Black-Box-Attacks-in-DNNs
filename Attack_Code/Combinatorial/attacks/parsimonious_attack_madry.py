# import cv2
import itertools
import math
import numpy as np
import tensorflow as tf
import time
from PIL import Image
import sys
import time
import cv2 

from Attack_Code.Combinatorial.attacks.local_search_helper_madry import LocalSearchHelper


def ParsimoniousAttack(loss_f, image, label, args, **kwargs):
    """Parsimonious attack using local search algorithm"""

    # Hyperparameter setting
    max_queries = args.max_evals
    epsilon = args.eps
    batch_size = args.batch_size
    no_hier = args.no_hier
    block_size = args.block_size
    no_hier = args.no_hier
    num_channels = args.num_channels
    dim_image = args.dim_image
    subspace_attack = args.subspace_attack
    subspace_dim = args.subspace_dimension


    def _perturb_image(width, height, image, noise):
            """Given an image and a noise, generate a perturbed image.
            First, resize the noise with the size of the image.
            Then, add the resized noise to the image.

            Args:
                image: numpy array of size [1, 299, 299, 3], an original image
                noise: numpy array of size [1, 256, 256, 3], a noise

            Returns:
                adv_iamge: numpy array of size [1, 299, 299, 3], an perturbed image
            """
            adv_image = image + cv2.resize(noise[0, ...], (width, height), interpolation=cv2.INTER_NEAREST)
            if width != 96:
                adv_image = np.clip(adv_image, -0.5, 0.5)
            else:
                adv_image = np.clip(adv_image, -1, 1)
            return np.array([adv_image])


    def _split_block(upper_left, lower_right, block_size, channels):
        """Split an image into a set of blocks.
        Note that a block consists of [upper_left, lower_right, channel]

        Args:
            upper_left: [x, y], the coordinate of the upper left of an image
            lower_right: [x, y], the coordinate of the lower right of an image
            block_size: int, the size of a block

        Returns:
            blocks: list, the set of blocks
        """
        blocks = []
        if not subspace_attack:
            xs = np.arange(upper_left[0], lower_right[0], block_size)
            ys = np.arange(upper_left[1], lower_right[1], block_size)
            for x, y in itertools.product(xs, ys):
                for c in range(channels):
                    blocks.append([[x, y], [x+block_size, y+block_size], c])
            return blocks
        else:
            x_position, y_position, c_position = cohordinates_most_variance(np.array([image]), subspace_dim)
            for i in range(len(x_position)):
                blocks.append([[x_position[i], y_position[i]], 
                               [x_position[i]+1, y_position[i]+1], 
                                c_position[i]])
            return blocks

  
    # Class variables
    width = image.shape[0]
    height = image.shape[1]
    channels = image.shape[2]
    adv_image = np.copy(image)
    num_queries = 0

    upper_left = [0, 0]
    if no_hier:
        lower_right = [width,height]
        block_size = 1
    else:
        lower_right = [256, 256]

    # Split an image into a set of blocks
    blocks = _split_block(upper_left, lower_right, block_size, num_channels)

    # Initialize a noise to -epsilon
    if not no_hier and not subspace_attack:
        noise = -epsilon*np.ones([1, 256, 256, channels], dtype=np.float32)
    else:
        noise = -epsilon*np.ones([1, width, height, channels], dtype=np.float32)


    # Construct a batch
    num_blocks = len(blocks)
    batch_size = batch_size if batch_size > 0 else num_blocks
    curr_order = np.random.permutation(num_blocks)
    loss = np.inf

    
    time_beg = time.time()
    time_end = time.time()
    initial_batch = 0

    internal_queries = 0

    # Main loop
    while True:
        # Run batch
        num_batches = int(math.ceil(num_blocks/batch_size))

        # print('We got ', num_blocks,' blocks and use batches of dimension ', batch_size)
        for i in range(initial_batch, num_batches):
            # print(i, num_batches)
            try:
                print("[STATS][L2] rate = {:.5g}, cost = {}, size = {}, loss = {:.5g}, time = {:.5g}".
                        format(i/num_batches, num_queries, block_size, loss, time_end-time_beg))
                sys.stdout.flush()
            except:
                print('could not print', loss,num_queries,i, num_batches, block_size)
            time_beg = time.time()
            # Pick a mini-batch
            bstart = i*batch_size
            bend = min(bstart + batch_size, num_blocks)
            blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
            # Run local search algorithm on the mini-batch

            noise, queries, loss, success = LocalSearchHelper(loss_f, image, noise,
                                blocks_batch, num_queries, label, args)
            time_end = time.time()
            num_queries += queries
            internal_queries += queries
            
            # Generate an adversarial image
            # print('Going for the perturbation')
            adv_image = _perturb_image(width, height, image, noise)
            # If query count exceeds the maximum queries, then return False
            
            if num_queries > max_queries:
                return adv_image[0], num_queries, [], success
            # If attack succeeds, return True
            if success:
                return adv_image[0], num_queries, [], success

        # If block size >= 2, then split the iamge into smaller blocks and reconstruct a batch
        
        if not no_hier and block_size >= 2 and block_size/256*dim_image>1 and not subspace_attack:
            print('CHANGING BLOCK SIZE')
            block_size //= 2
            blocks = _split_block(upper_left, lower_right, block_size, num_channels)
            num_blocks = len(blocks)
            batch_size = batch_size if batch_size > 0 else num_blocks
            curr_order = np.random.permutation(num_blocks)
        # Otherwise, shuffle the order of the batch
        else:
            curr_order = np.random.permutation(num_blocks)



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

    


