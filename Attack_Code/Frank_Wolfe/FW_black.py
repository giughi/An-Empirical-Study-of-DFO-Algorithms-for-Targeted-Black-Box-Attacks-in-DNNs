import time
from Attack_Code.Frank_Wolfe.utils import *
import numpy as np

class FW_black:
    def __init__(self, loss_f, att_iter=10000, grad_est_batch_size=25, eps=0.05,
                 lr=0.01, delta=0.01, sensing_type='gaussian', q_limit=50000, beta1=0.99):
        self.att_iter = att_iter
        self.grad_est_batch_size = grad_est_batch_size
        self.batch_size = 1  # Only support batch_size = 1 in black-box setting
        self.epsilon = eps
        self.clip_min = -0.5
        self.clip_max = 0.5
        self.lr = lr
        self.delta = delta
        self.sensing_type = sensing_type
        self.q_limit = q_limit
        self.beta1 = beta1
        self.loss_f = loss_f

    def single_batch_grad(self, img, lab):        
        # GRADIENT ESTIMATION GRAPH
        grad_estimates = []
        final_losses = []
        noise_pos = np.random.normal(size=(self.grad_est_batch_size,)+self.single_shape)
        if self.sensing_type == 'sphere':
            reduc_ind = list(range(1, len(self.shape)))
            noise_norm = np.sqrt(np.sum(np.square(noise_pos), reduction_indices=reduc_ind, keep_dims=True))
            noise_pos = noise_pos / noise_norm
            d = np.prod(self.single_shape)
            noise_pos = noise_pos * (d ** 0.5)
            noise = np.concatenate((noise_pos, -noise_pos), axis=0)
        elif self.sensing_type == 'gaussian':
            noise = np.concatenate((noise_pos, -noise_pos), axis=0)
        else:
            print ('Unknown Sensing Type')
            import sys
            sys.exit()
        grad_est_imgs = img + self.delta * noise
        # grad_est_labs = np.ones([self.grad_est_batch_size * 2, 1]) * lab
        grad_est_losses,_,_ = self.loss_f(grad_est_imgs)
        grad_est_losses_tiled = np.tile(np.reshape(grad_est_losses, (-1, 1, 1, 1)), (1,) + self.single_shape)
        grad_estimates.append(np.mean(grad_est_losses_tiled * noise, axis=0) / self.delta)
        final_losses.append(grad_est_losses)

        grad_estimate = np.mean(grad_estimates, axis=0)
        final_losses = np.concatenate(final_losses, axis=0)
        return final_losses, grad_estimate

    # GRADIENT ESTIMATION EVAL
    def get_grad_est(self, x, batch_lab, num_batches):
        losses = []
        grads = []
        for _ in range(num_batches):
            final_losses, grad_estimate = self.single_batch_grad(x,batch_lab)
            losses.append(final_losses)
            grads.append(grad_estimate)
        grads = np.array(grads)
        losses = np.array(losses)
        return losses.mean(), np.mean(grads, axis=0, keepdims=True)

    def attack(self, inputs, targets):
        """
        targets=scalar
        """
        image_size, num_channels = inputs.shape[2:]

        self.shape = (None, image_size, image_size, num_channels)
        self.single_shape = (image_size, image_size, num_channels)

        adv = np.copy(inputs)
        stop_query = 0
        stop_time = 0
        loss_init, pred_init, eval_adv = self.loss_f(inputs) # loss, predicted class, success?
        finished_mask = eval_adv
        succ_sum = finished_mask
        dist = 0
        print ("[L2] Init Loss : % 5.3f, Dist: % 5.3f," % (
            loss_init[0], dist), ' Finished: ', str(succ_sum))

        if succ_sum == len(inputs):
            return inputs, stop_query, _, finished_mask


        data = inputs[0]
        lab = targets
        ori = data
        x = data
        num_batches = 1
        m_t = np.zeros_like(data)

        last_ls = []
        hist_len = 5
        min_lr = 1e-3
        current_lr = self.lr
        start_decay = 0

        for iteration in range(self.att_iter):
            start_time = time.time()

            #checking if evaluations reached the limit
            stop_query += num_batches * self.grad_est_batch_size * 2

            if stop_query > self.q_limit:
                stop_query = self.q_limit
                return adv, stop_query, _, finished_mask

            # Get zeroth-order gradient estimates
            _, grad = self.get_grad_est(x, lab, num_batches)
            # momentum
            m_t = m_t * self.beta1 + grad * (1 - self.beta1)
            grad_normalized = grad_normalization(m_t)

            s_t = - self.epsilon * grad_normalized + ori
            d_t = s_t - x
            current_lr = self.lr if start_decay == 0 else self.lr / (iteration - start_decay + 1) ** 0.5
            new_x = x + current_lr * d_t
            new_x = np.clip(new_x, self.clip_min, self.clip_max)

            x = new_x
            stop_time += (time.time() - start_time)

            loss, pred, eval_adv = self.loss_f(x)

            last_ls.append(loss[0])
            last_ls = last_ls[-hist_len:]
            if last_ls[-1] > 0.999 * last_ls[0] and len(last_ls) == hist_len:
                if start_decay == 0:
                    start_decay = iteration - 1
                    print ("[log] start decaying lr")
                last_ls = []

            finished_mask = eval_adv

            if iteration % 10 == 0:
                dist = get_dist(x, ori)
                print ("[L2] Iter: %3d, Loss: %5.3f, Dist: %5.3f, Lr: %5.4f, Finished: %3d, Query: %3d"
                    % (iteration, loss[0], dist, current_lr, succ_sum, stop_query))

            if finished_mask:
                break

        adv = new_x

        dist = get_dist(x, ori)
        print ("[L2] End Loss : % 5.3f, Dist: % 5.3f, Finished: % 3d,  Query: % 3d " % (
            loss[0], dist, finished_mask, stop_query))

        return adv, stop_query, _, finished_mask