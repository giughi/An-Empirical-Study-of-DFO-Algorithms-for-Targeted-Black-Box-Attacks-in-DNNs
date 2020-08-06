"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""
import time
import random
import numpy as np
import tensorflow as tf   
import cv2


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)

def softmax_2(x):
    e_x = []
    for j in range(len(x)):
        e_x.append(np.exp(x[j] - np.max(x[j], axis=0)))
        e_x[-1] /= e_x[-1].sum(axis=0)
    return np.array(e_x)
# from Setups.Data_and_Model.setup_inception import ImageNet, InceptionModel



class GenAttack2(object):
    def mutation_op(self,  cur_pop, idx, step_noise=0.01, p=0.005):
        perturb_noise = 2*step_noise*np.random.random(cur_pop.shape) - step_noise
        mutated_pop = perturb_noise * (np.random.random(cur_pop.shape) < p) + cur_pop
        return mutated_pop


    def attack_step(self, idx, success, orig_copies, cur_noise, prev_elite, best_win_margin, cur_plateau_count, num_plateaus):
        if self.resize_dim:
            noise_resized = []
            for j in range(cur_noise.shape[0]):
                noise_resized.append(cv2.resize(cur_noise[j], (self.input_dim,self.input_dim), 
                                       interpolation=cv2.INTER_LINEAR))
            noise_resized = np.array(noise_resized)
        else:
            noise_resized = cur_noise
        noise_dim = self.resize_dim or self.input_dim
        cur_pop = np.clip(noise_resized, self.box_min, self.box_max)
        time_befor = time.time()
        pop_preds = self.model.predict(cur_pop)
        pop_preds = softmax_2(pop_preds)
        time_after = time.time()
        all_preds = np.argmax(pop_preds, axis=1)

        success_pop = (all_preds == self.target)
        success = np.max(success_pop, axis=0)

        target_scores = np.sum(self.tlab * pop_preds, axis=1)
        sum_others = np.sum((1-self.tlab)* pop_preds, axis=1)
        max_others = np.max((1-self.tlab)* pop_preds, axis=1)
        loss = -(np.log(sum_others+1e-10) - np.log(target_scores+1e-10))
        win_margin = np.max(pop_preds[:, self.target] - np.max(pop_preds, axis=1))
        
        new_best_win_margin, new_cur_plateau_count = (win_margin, 0) if (win_margin>
                    best_win_margin) else (best_win_margin, cur_plateau_count+1)

        plateau_threshold = 100 if win_margin>-0.40 else 300
        new_num_plateaus, new_cur_plateau_count = (num_plateaus+1, 0) if (
                    new_cur_plateau_count>plateau_threshold) else (num_plateaus, new_cur_plateau_count)

        if self.adaptive:
            step_noise =   np.maximum(self.alpha, 
                    0.4*np.power(0.9, new_num_plateaus))
            step_p =  1 if idx < 10 else np .maximum(
                                self.mutation_rate, 0.5*np.power(0.90, new_num_plateaus))
        else:
            step_noise = self.alpha
            step_p = self.mutation_rate

        step_temp = 0.1


        elite_idx = np.expand_dims(np.argmax(success_pop), axis=0) if (
                    success== 1) else np.expand_dims(np.argmax(loss, axis=0), axis=0)

        elite = cur_noise[elite_idx]
        select_probs = softmax(loss/ step_temp)
        parents = np.random.choice(np.arange(self.pop_size), 2*self.pop_size-2, p=select_probs)
        parent1 = cur_noise[parents[:self.pop_size-1]]
        parent2 = cur_noise[parents[self.pop_size-1:]]
        pp1 = select_probs[parents[:self.pop_size-1]]
        pp2 = select_probs[parents[self.pop_size-1:]]
        pp2 = pp2 / (pp1+pp2)
        pp2 = np.tile(np.expand_dims(np.expand_dims(np.expand_dims(pp2, 1), 
                                                    2), 
                                     3), 
                      (1, noise_dim, noise_dim,3))
        prob = np.random.random(parent1.shape)
        xover_prop = (prob > pp2)
        childs = parent1 * xover_prop + parent2 * (1-xover_prop)
        idx += 1
        if np.mod(idx,10):
            print(idx, np.min(loss), win_margin, step_p, step_noise, new_cur_plateau_count,time_after-time_befor)
       
        
        mutated_childs = self.mutation_op(childs, idx=idx, step_noise=self.eps*step_noise, p=step_p)
        new_pop = np.concatenate((mutated_childs, elite), axis=0)
        return idx, success, orig_copies, new_pop, np.reshape(elite,(noise_dim, noise_dim, 3)), new_best_win_margin, new_cur_plateau_count, new_num_plateaus


    def __init__(self, model, pop_size=6, mutation_rate=0.001,
            eps=0.15, max_evals=10000, alpha=0.20,
            resize_dim=None, adaptive=False, num_classes=1001, input_dim=299):
        self.eps = eps
        self.pop_size = pop_size
        self.model = model
        self.alpha = alpha
        self.max_evals = max_evals
        self.mutation_rate = mutation_rate
        self.resize_dim = np.min((resize_dim, input_dim))
        noise_dim = self.resize_dim
        self.adaptive = adaptive
        self.num_classes = num_classes
    
        # copies of original image
        self.pop_orig = np.zeros((self.pop_size, input_dim, input_dim, 3), dtype=np.float32)
        self.pop_noise = np.zeros((self.pop_size, noise_dim, noise_dim, 3), dtype=np.float32)

        self.target = 0
        self.init_success = 0
        self.i = 0

        # Variables to detect plateau
        self.best_win_margin = -1
        self.cur_plateau_count = 0
        self.num_plateaus = 0


    def cond(self, i, success, pop_orig, pop_noise, cur_elite, best_win_margin, cur_plateau_count, num_plateaus): 
        return ((i* (self.pop_size  - 1)+ 1 <= self.max_evals) and (success== 0))
        
        
    def attack_main(self):
        i, success, pop_orig, pop_noise, cur_elite,  best_win_margin, cur_plateau_count, num_plateaus = [self.i, 
                self.init_success, self.pop_orig, self.pop_noise, self.pop_noise[0], self.best_win_margin, 
                self.cur_plateau_count, self.num_plateaus]
        cond = self.cond(i, success, pop_orig, pop_noise, cur_elite, best_win_margin, cur_plateau_count, num_plateaus)
        
        while(cond):
            (i, success, pop_orig, pop_noise, cur_elite, best_win_margin, cur_plateau_count, 
                    num_plateaus) = self.attack_step(i, success, pop_orig, pop_noise, 
                                            cur_elite, best_win_margin, cur_plateau_count, num_plateaus)
            
            cond = self.cond(i, success, pop_orig, pop_noise, cur_elite, best_win_margin, cur_plateau_count, num_plateaus)
        
        return i, success, pop_orig, pop_noise, cur_elite,  best_win_margin, cur_plateau_count, num_plateaus


    def attack(self, input_img, target_label): 
        self.input_img = input_img
        self.target = target_label
        self.input_dim = input_img.shape[1]
        self.box_min = np.tile(np.maximum(self.input_img-self.eps, -0.5), (self.pop_size, 1,1,1))
        self.box_max = np.tile(np.minimum(self.input_img+self.eps, 0.5), (self.pop_size, 1,1,1))
        self.tlab = np.eye(self.num_classes)[self.target]

        (num_steps, success,  copies, final_pop, adv_noise,_,_,_) = self.attack_main()
        if self.resize_dim:
            adv_img = np.clip(np.expand_dims(input_img, axis=0)+np.expand_dims(cv2.resize(
                                    adv_noise, (self.input_dim,self.input_dim), 
                                    interpolation=cv2.INTER_LINEAR), axis=0),
                                self.box_min[0:1], self.box_max[0:1])
        else:
            adv_img = np.clip(np.expand_dims(input_img, axis=0)+np.expand_dims(adv_noise, axis=0),
                self.box_min[0:1], self.box_max[0:1])

        # Number of queries = NUM_STEPS * (POP_SIZE -1 ) + 1
        # We subtract 1 from pop_size, because we use elite mechanism, so one population 
        # member is copied from previous generation and no need to re-evaluate it.
        # The first population is an exception, therefore we add 1 to have total sum.
        query_count = num_steps * (self.pop_size  - 1)+ 1
        
        # query_count = num_steps * (self.pop_size  - 1)+ 1
        if success:
            return  adv_img[0], query_count, _, success
        else:
            query_count = num_steps * (self.pop_size  - 1)+ 1
            return  adv_img[0], query_count, _, success
