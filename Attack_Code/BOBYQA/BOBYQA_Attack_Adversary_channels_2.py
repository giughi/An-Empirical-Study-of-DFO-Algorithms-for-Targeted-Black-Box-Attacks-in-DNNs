from __future__ import print_function

import sys
# import os
import tensorflow as tf
import numpy as np
import time

import pybobyqa
import pandas as pd

import cv2
# Initialisation Coefficients

MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be


class Objfun(object):
    def __init__(self, objfun):
        self._objfun = objfun
        self.nf = 0
        self.xs = []
        self.fs = []
        self.ds = []

    def __call__(self, x):
        self.nf += 1
        self.xs.append(x.copy())
        f, _, d = self._objfun(x)
        self.fs.append(f[0])
        self.ds.append(d[0])
        return f[0]

    def get_summary(self, with_xs=False):
        results = {}
        if with_xs:
            results['xvals'] = self.xs
        results['fvals'] = self.fs
        results['dvals'] = self.ds
        results['nf'] = self.nf
        results['neval'] = np.arange(1, self.nf+1)  # start from 1
        return pd.DataFrame.from_dict(results)

    def reset(self):
        self.nf = 0
        self.xs = []
        self.fs = []


def vec2modMatRand3(c, indice, var, depend, b, a, overshoot):
    """
    With this function we want to mantain the optiomisation domain
    centered.RandMatr
    """
    temp = var.copy().reshape(-1, )
    n = len(indice)
    for i in range(n):
        indices = finding_indices(depend.reshape(-1, ), indice[i])
        if overshoot:
            temp[indices] += c[i]*np.max((b-a).reshape(-1, )[indices])/2 + (b+a).reshape(-1, )[indices]/2
        else:
            temp[indices] += c[i]*((b-a).reshape(-1, )[indices])/2 + (b+a).reshape(-1, )[indices]/2
    # we have to clip the values to the boundaries
    temp = np.minimum(b.reshape(-1, ), temp.reshape(-1, ))
    temp = np.maximum(a.reshape(-1, ), temp)
    # temp[b-temp<temp-a] = b[b-temp<temp-a]
    # temp[b-temp>temp-a] = a[b-temp>temp-a]
    temp = temp.reshape(var.shape)
    return temp

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
    nn_up = np.floor(img_size/n)
    for i in range(n):
        partition.append(int(i*nn_up))
    partition.append(img_size)
    # print(partition)
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


def finding_indices(dependency, index):
    """
    This returns a boolean tensor with the same shape of dependency  with the elements 
    that are equal to the scalar index.
    """
    return dependency == index


class BlackBox_BOBYQA:
    def __init__(self, loss_f, batch_size=1, interpolation='block', n_channels_input=3,
                 print_every=100, use_resize=False, eps=0.15, max_eval=1e5, 
                 over='over', rounding=True, max_f=1.2, subspace_attack=False,
                 subspace_dim=None):
        """
        The BOBYQA attack.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of gradient evaluations to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        """
        self.loss_f = loss_f
        self.target = None
        self.print_every = print_every
        self.batch_size = batch_size
        self.subspace_attack = subspace_attack
        self.subspace_dim = subspace_dim

        self.small_x = None
        self.small_y = None
        
        self.eps = eps
        self.use_resize = use_resize
        self.max_eval = max_eval
        self.max_f = max_f
        self.rounding = rounding
        self.n_channels_input = n_channels_input
  
        if over == 'over':
            self.overshoot=True
        elif over == 'linear':
            self.overshoot=False
        else:
            print('ERRROR, NOT RIGHT CLASSIFICATION TERM')
        self.l  =  0 
        
    def resize_img(self):
        small_single_shape = (self.small_x, self.small_y, self.small_channels)
        self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.small_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size
                
    def blackbox_optimizer(self, iteration, ord_domain, super_dependency, img, img0):
        # build new inputs, based on current variable value
        var = 0*np.array([img])
        NN = self.var_list.size
        if self.subspace_attack:
            NN = len(ord_domain)
    
        if len(ord_domain)<self.batch_size:
            nn = len(ord_domain)
        else:
            nn = self.batch_size
        # We choose the elements of ord_domain that are inherent to the step. 
        # So it is already limited to the variable's dimension
        if (iteration+1)*nn <= NN:
            var_indice = ord_domain[iteration*nn: (iteration+1)*nn]
        elif self.subspace_attack:
            lower  = np.mod((iteration)*nn,NN)
            upper  = np.mod((iteration+1)*nn,NN)
            if upper > lower:
                # if the right and left bound are not inverted by the mod operation
                var_indice = ord_domain[lower: upper]
            else:
                var_indice = ord_domain[lower:]
                var_indice = np.concatenate((var_indice, ord_domain[:upper]))
        else:
            var_indice = ord_domain[list(range(iteration*nn, NN))]
            nn = NN - iteration*nn
        indice = self.var_list[var_indice]
        x_o = np.zeros(nn,)
        # Define the bounds of the optimisation variable
        if self.use_resize:
            a = -np.ones((nn,))
            b = np.ones((nn,))
            # find the initial condition that identifies the previous perturbation
            for i in range(nn):
                indices = finding_indices(super_dependency.reshape(-1, ), indice[i])
                up = self.modifier_up[indices] 
                down = self.modifier_down[indices] 
                max_ind = np.argmax(up-down)
                xs =  np.divide( -(up+down),
                                (up-down))
                x_o[i] = np.clip(xs[max_ind],-1,1)
        else:
            b = self.modifier_up[indice]
            a = self.modifier_down[indice]
        bb = self.modifier_up
        aa = self.modifier_down
        # define the loss function
        opt_fun = Objfun(lambda c: self.loss_f(img, vec2modMatRand3(c, indice, var, super_dependency, bb, aa,
                                                               self.overshoot), only_loss=False))
        initial_loss = opt_fun(x_o)
        if np.abs(initial_loss - self.l)>10e-6:
            print('[WARNING] Rebuilt intial vecotr has a loss different by', initial_loss-self.l)
        user_params = {'init.random_initial_directions':False, 
                       'init.random_directions_make_orthogonal':False}
        soln = pybobyqa.solve(opt_fun, x_o, rhobeg=np.min(b-a)/3,
                              bounds=(a, b), maxfun=nn*self.max_f,
                              rhoend=np.min(b-a)/6,
                              npt=nn+1, scaling_within_bounds=False,
                              user_params=user_params)
        summary = opt_fun.get_summary(with_xs=False)
        minimiser = np.min(summary['fvals'])
        distances = np.array(summary['dvals'])
        early_discovery=0
        if np.any(distances<=0):
            # not counting the evaluations done after having found an example
            # for which distance <=0 i.e. an adversarial ex was found.
            early_discovery = (np.max(summary['neval']) - 
                               summary['neval'][np.where(distances<=0)[0][0]] + 2)
            print('Early Discover made at ', early_discovery)

        real_oe = self.loss_f(img, vec2modMatRand3(soln.x, indice, var, super_dependency, bb, aa,
                                                    self.overshoot), only_loss=True)
        if (minimiser != real_oe) and (initial_loss>minimiser):
            print('[WARNING] BOBYQA returns not the minimal samples function.')
        evaluations = soln.nf

        nimgs = vec2modMatRand3(soln.x, indice, var, super_dependency, bb, aa, self.overshoot)
        distance = self.loss_f(img,nimgs, only_loss=True)
        if self.rounding:
            # checking if the rounded image works better
            nimg2 = nimgs.copy()
            nimg2.reshape(-1,)[bb-nimgs.reshape(-1,)<nimgs.reshape(-1,)-aa] = bb[bb-nimgs.reshape(-1,)<nimgs.reshape(-1,)-aa]
            nimg2.reshape(-1,)[bb-nimgs.reshape(-1,)>nimgs.reshape(-1,)-aa] = aa[bb-nimgs.reshape(-1,)>nimgs.reshape(-1,)-aa]
            distance2 = self.loss_f(img, nimg2, only_loss=True)
            if distance2 < distance:
                print('[WARNING][L5] Using rounded perturbation to the domain')
                return distance2[0], evaluations + 2 - early_discovery, nimg2, summary
        return distance[0], evaluations + 2 - early_discovery, nimgs, summary

    def attack_batch(self, img, lab):
        """
        Run the attack on a batch of images and labels.
        """
        self.target = np.argmax(lab)
        
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x = np.argmax(x)
            return x == y

        # remove the extra batch dimension
        if len(img.shape) == 4:
            img = img[0]
        if len(lab.shape) == 2:
            lab = lab[0]
        self.image_size,_, self.num_channels = img.shape
        self.var_size_b = self.image_size **2 * self.num_channels
        self.modifier_up = np.zeros(self.var_size_b, dtype=np.float32)
        self.modifier_down = np.zeros(self.var_size_b, dtype=np.float32)
        img0 = img

        # define the dimension of the perturbation that have to be considered
        init_width = self.n_channels_input
        if self.use_resize and not self.subspace_attack:
            dimen = np.power(2,np.arange(1, np.ceil(np.log2(self.image_size)), 
                                            dtype=np.int)
                            )
            steps = [0]
            width = [init_width]
            for d in dimen[:-1]:
                n_var = d**2 * width[-1]
                n_step = np.ceil(n_var/self.batch_size)
                steps.append(steps[-1] + n_step)
                width.append(init_width)
            steps.append(steps[-1]*2)
            dimen = np.append(dimen,self.image_size)
            width.append(init_width)
            steps.append(steps[-1]*2)
            dimen = np.append(dimen,self.image_size)
            width.append(3)
        else:
            steps = [0]
            dimen = [self.image_size]
            width = [3]

        # initialise the modifier
        self.small_x = dimen[0]
        self.small_y = dimen[0]
        self.small_channels = width[0]
        self.resize_img()

        # intialise the parameters
        eval_costs = 0       
        # ord_domain = np.random.choice(self.var_list.size, self.var_list.size, 
        #                               replace=False, p=self.sample_prob)
        iteration_scale = 0
        iteration_domain = 0
        step = -1
        global_summary = []
        adv = 0 * img

        # initialise some dummy variable for the computations later on
        zz = np.zeros((self.image_size, self.image_size, self.num_channels))
        ub = 0.5*np.ones((self.image_size, self.image_size, self.num_channels))
        lb = -ub

        # Run the attack steps
        while eval_costs<self.max_eval:
            step+=1
            # print the status of the attack every self.print_every iterations
            if step % self.print_every == 0:
                loss, output, distance= self.loss_f(img, np.array([adv]))
                print("[STATS][L2] iter = {}, cost = {}, iter_sc = {:.3f},"
                      "iter_do = {:.3f}, size = {}, loss = {:.5g}, loss_f = {:.5g}, "
                      "maxel = {}".format(step, eval_costs, iteration_scale, 
                                          iteration_domain, self.real_modifier.shape, 
                                          distance[0], loss[0],
                                          np.argmax(output[0])))
                sys.stdout.flush()
                l = loss[0]
                self.l = l

            # Define the upper and lower bounds on the perturbation pixelwise    
            self.modifier_up = np.maximum(np.minimum(- (img.reshape(-1,) - img0.reshape(-1,)) + self.eps,
                                                        ub.reshape(-1,) - img.reshape(-1,)),
                                            zz.reshape(-1,))
            self.modifier_down = np.minimum(np.maximum(- (img.reshape(-1,) - img0.reshape(-1,)) - self.eps,
                                                        - img.reshape(-1,) + lb.reshape(-1,)),
                                            zz.reshape(-1,))
            
            iteration_scale += 1
            iteration_domain = np.mod(iteration_scale, (self.use_var_len//self.batch_size + 1))

            force_renew = False
            # if we have finished spanning the domain
            if iteration_domain == 0:
                force_renew=True
            # if self.use_resize:
            if step in steps:
                idx = steps.index(step)
                self.small_x = dimen[idx]
                self.small_y = dimen[idx]
                self.small_channels = width[idx]
                self.resize_img()
                iteration_scale = 0
                iteration_domain = 0
                force_renew = True

            if self.subspace_attack:
                # we need to reshufle the subspace every time we have gone through it
                if np.round(step*self.batch_size/self.subspace_dim) != np.round((step-1)*self.batch_size/self.subspace_dim):
                    print('reshuffling the order')
                    force_renew = True

            if  force_renew:  
                super_dependency = matr_subregions_division(self.image_size, self.num_channels,
                                                            self.small_x, self.small_channels
                                                            )
                prob = image_region_importance(cv2.resize(img, (self.small_x, self.small_y), 
                                                          interpolation=cv2.INTER_LINEAR),
                                              self.small_channels).reshape(-1,)
                if not self.subspace_attack:
                    ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p=prob)
                else:
                    top_n_indices = np.argsort(prob)[-self.subspace_dim:]
                    ord_domain = np.random.choice(top_n_indices, self.subspace_dim, replace=False, 
                                                  p=prob[top_n_indices]/np.sum(prob[top_n_indices]))
            l, evaluations, nimg, summary = self.blackbox_optimizer(iteration_domain,
                                                                    ord_domain,
                                                                    super_dependency,
                                                                    img,img0)

            self.l = l

            global_summary.append(summary)
            
            adv = nimg[0]
            
            adv = adv.reshape((self.image_size, self.image_size, self.num_channels))
            img = img + adv
            eval_costs += evaluations

            adv= 0*adv

            loss, output, distance = self.loss_f(img, np.array([adv]))
            score = output[0]

            if compare(score, np.argmax(lab)):
                print("[STATS][L3](First valid attack found!) iter = {}, cost = {}".format(step, eval_costs))
                sys.stdout.flush()
                o_bestl2 = l
                o_bestattack = img
            
            if distance[0] <= 0:
                print("[STATS][L3]Early Stopping becuase minimum reached")
                if len(o_bestattack.shape) == 3:
                    o_bestattack = o_bestattack.reshape((1,) + o_bestattack.shape)
                return o_bestattack, eval_costs, global_summary, True

        print('[STATS][L3]The algorithm did not converge')
        if len(img.shape) == 3:
            img = img.reshape((1,) + img.shape)
        return img, eval_costs, global_summary, False 