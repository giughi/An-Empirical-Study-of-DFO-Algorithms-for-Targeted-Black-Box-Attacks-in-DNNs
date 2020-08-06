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


def vec2modMatRand3(c, Random_Matrix, b, a, var):
    """
    With this function we want to mantain the optiomisation domain
    centered.RandMatr
    """
    temp = np.matmul(Random_Matrix,c).reshape(-1,)
    # we have to clip the values to the boundaries
    temp = np.minimum(b.reshape(-1, ), temp.reshape(-1, ))
    temp = np.maximum(a.reshape(-1, ), temp)
    # temp[b-temp<temp-a] = b[b-temp<temp-a]
    # temp[b-temp>temp-a] = a[b-temp>temp-a]
    temp = temp.reshape(var.shape)
    return temp

class BlackBox_BOBYQA_random_proj(object):
    def __init__(self, loss_f, batch_size=1,  n_channels_input=3,
                 print_every=100, eps=0.15, max_eval=1e5, 
                 max_f=1.2, subspace_attack=False,
                 subspace_dim=None, delta=0.01):
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
        
        self.delta = delta

        self.eps = eps
        self.max_eval = max_eval
        self.max_f = max_f
        self.n_channels_input = n_channels_input
  
        self.l  =  0 
        
    def blackbox_optimizer(self, img, img0):
        # build new inputs, based on current variable value
        var = 0*np.array([img])
        nn = self.batch_size
        x_o = np.zeros(nn,)
        Random_Matrix = np.random.normal(size=(self.var_size_b,nn))*self.delta
        # Define the bounds of the optimisation variable
        a = -np.ones((nn,))
        b = np.ones((nn,))

        bb = self.modifier_up
        aa = self.modifier_down
        # define the loss function
        opt_fun = Objfun(lambda c: self.loss_f(img, vec2modMatRand3(c, Random_Matrix, bb, aa, var), 
                                               only_loss=False)
                        )
        initial_loss = opt_fun(x_o)
        
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

        real_oe = self.loss_f(img, vec2modMatRand3(soln.x, Random_Matrix, bb, aa, var), only_loss=True)
        if (minimiser != real_oe) and (initial_loss>minimiser):
            print('[WARNING] BOBYQA returns not the minimal samples function.')
        evaluations = soln.nf

        nimgs = vec2modMatRand3(soln.x, Random_Matrix, bb, aa, var)
        distance = self.loss_f(img,nimgs, only_loss=True)
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


        # intialise the parameters
        eval_costs = 0       
        # ord_domain = np.random.choice(self.var_list.size, self.var_list.size, 
        #                               replace=False, p=self.sample_prob)
        iteration_scale = 0
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
                print("[STATS][L2] iter = {}, cost = {}, loss = {:.5g}, loss_f = {:.5g}, "
                      "maxel = {}".format(step, eval_costs,
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
            
           
                        
            l, evaluations, nimg, summary = self.blackbox_optimizer(img,img0)

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