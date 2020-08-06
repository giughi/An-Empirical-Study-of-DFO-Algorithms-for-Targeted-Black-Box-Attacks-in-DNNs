# import cv2
import heapq
import math
import numpy as np
import sys
import tensorflow as tf
import time
from PIL import Image
import pandas as pd
import cv2
        
def LocalSearchHelper(loss_f, image, noise, blocks, tot_num_queries, label, args, **kwargs):
    """A helper for local search algorithm.
    Note that since heapq library only supports min heap, we flip the sign of loss function.
    """

    # Hyperparameter setting
    epsilon = args.eps
    max_iters = args.max_iters
    targeted = args.targeted
    max_queries = args.max_evals
    no_hier = args.no_hier
    subspace_attack = args.subspace_attack
    subspace_dim = args.subspace_dimension
    # Network setting

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

    def _flip_noise(noise, block):
        """Flip the sign of perturbation on a block.
        Args:
          noise: numpy array of size [1, 256, 256, 3], a noise
          block: [upper_left, lower_right, channel], a block

        Returns:
          noise: numpy array of size [1, 256, 256, 3], an updated noise
        """
        noise_new = np.copy(noise)
        upper_left, lower_right, channel = block
        noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
        return noise_new


    # Class variables
    width = image.shape[0]
    height = image.shape[1]
    channels = image.shape[2]
    # Local variables
    priority_queue = []
    num_queries = 0

    # Check if a block is in the working set or not
    A = np.zeros([len(blocks)], np.int32)
    for i, block in enumerate(blocks):
        upper_left, _, channel = block
        x = upper_left[0]
        y = upper_left[1]
        # If the sign of perturbation on the block is positive,
        # which means the block is in the working set, then set A to 1
        if noise[0, x, y, channel] > 0:
            A[i] = 1
    
    image_batch = _perturb_image(width, height, image, noise)
    losses_int, preds_int = loss_f(image_batch)
    
    num_queries += 1
    curr_loss = losses_int[0]

    # Early stopping
    if targeted:
        if preds_int == label:
            return noise, num_queries, curr_loss, True
    else:
        if preds_int != label:
            return noise, num_queries, curr_loss, True
    # Main loop
    for iteration in range(max_iters):
        indices,  = np.where(A==0)

        batch_size = 100
        num_batches = int(math.ceil(len(indices)/batch_size))
        # time_beg_adding_nodes = time.time()
        for ibatch in range(num_batches):
            bstart = ibatch * batch_size
            bend = min(bstart + batch_size, len(indices))

            image_batch = np.zeros([bend-bstart, width, height, channels], np.float32)
            if no_hier or subspace_attack:
                noise_batch = np.zeros([bend-bstart, width, height, channels], np.float32)
            else:
                noise_batch = np.zeros([bend-bstart, 256, 256, channels], np.float32)
            label_batch = np.tile(label, bend-bstart)

            for i, idx in enumerate(indices[bstart:bend]):
                noise_batch[i:i+1, ...] = _flip_noise(noise, blocks[idx])
                image_batch[i:i+1, ...] = _perturb_image(width, height, image, noise_batch[i:i+1, ...])

            losses_int, preds_int = loss_f(image_batch)
            # print(losses_int.shape, image_batch.shape)
            # Early stopping
            success_indices,  = np.where(preds_int == label) if targeted else np.where(preds_int != label)
            if len(success_indices) > 0:
                noise[0, ...] = noise_batch[success_indices[0], ...]
                curr_loss = losses_int[success_indices[0]]
                num_queries += success_indices[0] + 1
                print('Successfull at the lazy greedy insert')
                return noise, num_queries, curr_loss, True
            num_queries += bend-bstart

            if tot_num_queries+num_queries>max_queries:
                noise[0, ...] = noise_batch[0, ...]
                curr_loss = losses_int[0]
                return noise, num_queries, curr_loss, False

            # Push into the priority queue
            for i in range(bend-bstart):
                idx = indices[bstart+i]
                margin = losses_int[i]-curr_loss
                heapq.heappush(priority_queue, (margin, idx))
        # time_end_adding_nodes = time.time()
        # times[3] = time_end_adding_nodes - time_beg_adding_nodes
        # Pick the best element and insert it into the working set
        if len(priority_queue) > 0:
            best_margin, best_idx = heapq.heappop(priority_queue)
            curr_loss += best_margin
            noise = _flip_noise(noise, blocks[best_idx])
            A[best_idx] = 1
        # time_beg_working_set = time.time()
        # Add elements into the working set
        while len(priority_queue) > 0:
            # Pick the best element
            cand_margin, cand_idx = heapq.heappop(priority_queue)

            # Re-evalulate the element
            image_batch = _perturb_image(width, height, image, _flip_noise(noise, blocks[cand_idx]))
            label_batch = np.copy(label)

            losses_int, preds_int = loss_f(image_batch)

            num_queries += 1
            margin = losses_int[0]-curr_loss

            # If the cardinality has not changed, add the element
            if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
                # If there is no element that has negative margin, then break
                if margin > 0:
                    break
                # Update the noise
                curr_loss = losses_int[0]
                noise = _flip_noise(noise, blocks[cand_idx])
                A[cand_idx] = 1
                # Early stopping
                if targeted:
                    if preds_int == label:
                        return noise, num_queries, curr_loss, True
                else:
                    if preds_int != label:
                        return noise, num_queries, curr_loss, True
            # If the cardinality has changed, push the element into the priority queue
            else:
                heapq.heappush(priority_queue, (margin, cand_idx))
        priority_queue = []
        # Lazy greedy delete
        indices,  = np.where(A==1)

        batch_size = 100
        num_batches = int(math.ceil(len(indices)/batch_size))
        # time_beg_delete_set = time.time()
        for ibatch in range(num_batches):
            bstart = ibatch * batch_size
            bend = min(bstart + batch_size, len(indices))
            image_batch = np.zeros([bend-bstart, width, height, channels], np.float32)
            if no_hier or subspace_attack:
                noise_batch = np.zeros([bend-bstart, width, height, channels], np.float32)
            else:
                noise_batch = np.zeros([bend-bstart, 256, 256, channels], np.float32)
            label_batch = np.tile(label, bend-bstart)

            for i, idx in enumerate(indices[bstart:bend]):
                noise_batch[i:i+1, ...] = _flip_noise(noise, blocks[idx])
                image_batch[i:i+1, ...] = _perturb_image(width, height, image, noise_batch[i:i+1, ...])

            losses_int, preds_int = loss_f(image_batch)


            # Early stopping
            success_indices,  = np.where(preds_int == label) if targeted else np.where(preds_int != label)
            if len(success_indices) > 0:
                noise[0, ...] = noise_batch[success_indices[0], ...]
                curr_loss = losses_int[success_indices[0]]
                num_queries += success_indices[0] + 1
                return noise, num_queries, curr_loss, True
            num_queries += bend-bstart

            if tot_num_queries+num_queries>max_queries:
                noise[0, ...] = noise_batch[0, ...]
                curr_loss = losses_int[0]
                return noise, num_queries, curr_loss, False

            # Push into the priority queue
            for i in range(bend-bstart):
                idx = indices[bstart+i]
                margin = losses_int[i]-curr_loss
                heapq.heappush(priority_queue, (margin, idx))
        # Pick the best element and remove it from the working set
        if len(priority_queue) > 0:
            best_margin, best_idx = heapq.heappop(priority_queue)
            curr_loss += best_margin
            noise = _flip_noise(noise, blocks[best_idx])
            A[best_idx] = 0

        # Delete elements into the working set
        # time_beg_deleting_set = time.time()
        while len(priority_queue) > 0:
            # pick the best element
            cand_margin, cand_idx = heapq.heappop(priority_queue)

            # Re-evalulate the element
            image_batch = _perturb_image(width, height,image, _flip_noise(noise, blocks[cand_idx]))
            label_batch = np.copy(label)

            losses_int, preds_int = loss_f(image_batch)

            num_queries += 1
            margin = losses_int[0]-curr_loss

            # If the cardinality has not changed, remove the element
            if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
                # If there is no element that has negative margin, then break
                if margin >= 0:
                    break
                # Update the noise
                curr_loss = losses_int[0]
                noise = _flip_noise(noise, blocks[cand_idx])
                A[cand_idx] = 0
                # Early stopping
                if targeted:
                    if preds_int == label:
                        return noise, num_queries, curr_loss, True
                else:
                    if preds_int != label:
                        return noise, num_queries, curr_loss, True
            # If the cardinality has changed, push the element into the priority queue
            else:
                heapq.heappush(priority_queue, (margin, cand_idx))
        priority_queue = []
    return noise, num_queries, curr_loss, False
