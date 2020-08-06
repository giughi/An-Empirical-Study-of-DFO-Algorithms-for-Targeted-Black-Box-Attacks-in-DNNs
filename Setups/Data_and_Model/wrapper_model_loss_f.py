"""
This scripts allows to wrap the different instances of models and loss functions
for the different attacks
"""

import numpy as np
import torch as ch
# Patch for single output 
def patch_single_output(x, single_output):
    if single_output:
        return x,0
    return x

#Models

class Model_Class_combi():
    def __init__(self, model,single_output):
        self.model = model
        self.single_output = single_output

    def predict(self, img):
        img = np.moveaxis(img,2,0)
        img = ch.tensor([img]).float().cuda()
        logit,_ = patch_single_output(self.model(img+0.5), self.single_output)
        return logit.cpu().detach().numpy()

class Model_Class_boby():
    def __init__(self, model,single_output):
        self.model = model
        self.single_output = single_output

    def predict(self, img):
        img = np.moveaxis(img,3,1)
        img = ch.from_numpy(img).to(device='cuda',dtype=ch.float)
        logit,_ = patch_single_output(self.model(img+0.5), self.single_output)
        return logit.cpu().detach().numpy()

class Model_Class_square():
    def __init__(self, model,single_output):
        self.model = model
        self.single_output = single_output

    def predict(self, img):
        img = np.moveaxis(img,3,1)
        img = ch.tensor(img).float().cuda()
        logit,_ =patch_single_output(self.model(img+0.5), self.single_output)
        return logit[0].cpu().detach().numpy()
    
    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            probs = softmax_square(logits)
            preds_correct_class = (probs * y).sum(0, keepdims=True)
            diff = preds_correct_class - probs  
            diff[y] = np.inf  
            margin = diff.min(0, keepdims=True)
            loss_ = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = softmax_square(logits)
            loss_ = -np.log((probs*y).sum()+1e-10) + np.log(np.sum(probs)- (probs*y).sum() + 1e-10)
            loss_ = loss_ * -1 if not targeted else loss_
        else:
            raise ValueError('Wrong loss.')
        return loss_.flatten()

class Model_Class_combi_cpu():
    def __init__(self, model,single_output):
        self.model = model
        self.single_output = single_output

    def predict(self, img):
        img = np.moveaxis(img,2,0)
        img = ch.tensor([img]).float()
        logit,_ = patch_single_output(self.model(img+0.5), self.single_output)
        return logit.detach().numpy()

class Model_Class_boby_cpu():
    def __init__(self, model,single_output):
        self.model = model
        self.single_output = single_output

    def predict(self, img):
        img = np.moveaxis(img,3,1)
        img = ch.tensor(img).float()
        logit,_ = patch_single_output(self.model(img+0.5), self.single_output)
        return logit.detach().numpy()

class Model_Class_square_cpu():
    def __init__(self, model,single_output):
        self.model = model
        self.single_output = single_output

    def predict(self, img):
        img = np.moveaxis(img,3,1)
        img = ch.tensor(img).float()
        logit,_ =patch_single_output(self.model(img+0.5), self.single_output)
        return logit[0].detach().numpy()
    
    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            probs = softmax_square(logits)
            preds_correct_class = (probs * y).sum(0, keepdims=True)
            diff = preds_correct_class - probs  
            diff[y] = np.inf  
            margin = diff.min(0, keepdims=True)
            loss_ = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = softmax_square(logits)
            loss_ = -np.log((probs*y).sum()+1e-10) + np.log(np.sum(probs)- (probs*y).sum() + 1e-10)
            loss_ = loss_ * -1 if not targeted else loss_
        else:
            raise ValueError('Wrong loss.')
        return loss_.flatten()

class Model_Class_gene_cpu():
    def __init__(self, model,single_output):
        self.model = model
        self.single_output = single_output

    def predict(self, img):
        img = np.moveaxis(img,3,1)
        img = ch.tensor(img).float()
        logit,_ = patch_single_output(self.model(img+0.5), self.single_output)
        return logit.detach().numpy()

class Model_Class_FW_cpu():
    def __init__(self, model,single_output):
        self.model = model
        self.single_output = single_output

    def predict(self, img):
        img = np.moveaxis(img,3,1)
        img = ch.tensor(img).float()
        logit,_ = patch_single_output(self.model(img+0.5), self.single_output)
        return logit.detach().numpy()

def softmax_square(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

def wrapper_model(model, attack, single_output, cuda=False):
    if cuda:
        if attack =='boby':
            return Model_Class_boby(model,single_output)
        if attack =='boby_random':
            return Model_Class_boby(model,single_output)
        elif attack == 'combi':
            return Model_Class_combi(model,single_output)
        elif attack == 'square':
            return Model_Class_square(model,single_output)
        elif attack == 'gene':
            return Model_Class_boby(model,single_output)
        elif attack == 'FW':
            return Model_Class_boby(model,single_output)
    else:
        if attack =='boby':
            return Model_Class_boby_cpu(model,single_output)
        elif attack == 'combi':
            return Model_Class_combi_cpu(model,single_output)
        elif attack == 'square':
            return Model_Class_square_cpu(model,single_output)
        elif attack == 'gene':
            return Model_Class_gene_cpu(model,single_output)
        elif attack == 'FW':
            return Model_Class_FW_cpu(model,single_output)
            

# Loss functions

def loss_func_combi(img,targets, model):
    # global number_of_iterations
    nn= len(img)
    if nn==299:
        img = [img]
        nn = 1
    lll = []
    preds_l = []
    for i in range(nn):
        # number_of_iterations += 1
        # print('number_of_iterations', number_of_iterations)
        logits_ = model.predict(img[i])
        probs_ = softmax(logits_[0])
        indices = np.argmax(targets[0])
        lll.append(-np.log(probs_[indices] + 1e-10) + np.log(np.sum(probs_) 
                                                                - probs_[indices] + 1e-10))
        preds_l.append(np.argmax(logits_[0]))
    return lll, preds_l


def loss_func_boby(targets, model, img, pert, only_loss=False):
    nn= len(pert)
    if nn==299:
        img = [img]
        nn = 1
    lll = []
    preds_l = []
    distances = []
    
    for i in range(nn):
        logits_ = model.predict([img + pert[i]])
        probs_ = softmax(logits_[0])
        indices = np.argmax(targets)
        lll.append(-np.log(probs_[indices] + 1e-10) + np.log(np.sum(probs_)
                                                            - probs_[indices] + 1e-10))
        preds_l.append(np.argmax(logits_[0]))
        distances.append(np.max(probs_)-probs_[indices])
    
    if only_loss:
        return lll
    else:
        return lll, logits_, distances

def loss_func_FW(img, targets, model):
    logits_ = model.predict(img)
    lll = []
    preds_l = []
    successes = []
    for i in range(logits_.shape[0]):
        probs_ = softmax(logits_[i])
        indices = np.argmax(targets[0])
        lll.append(-np.log(probs_[indices] + 1e-10) + np.log(np.sum(probs_) 
                                                                - probs_[indices] + 1e-10))
        preds_l.append(np.argmax(logits_[i]))
        successes.append(np.argmax(logits_[i])==indices)
    return lll, preds_l, successes[0]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class wrapper_loss():
    def __init__(self, attack,targets,model):
        if attack =='boby':
            self.loss = lambda x,y,**kw: loss_func_boby(targets,model, x,y,**kw)
        if attack =='boby_random':
            self.loss = lambda x,y,**kw: loss_func_boby(targets,model, x,y,**kw)
        elif attack == 'combi':
            self.loss = lambda x: loss_func_combi(x,targets,model)
        elif attack == 'square':
            self.loss = None
        elif attack == 'gene':
            self.loss = None
        elif attack == 'FW':
            self.loss = lambda x: loss_func_FW(x,targets,model)

    def __call__(self, *args, **kw):
        return self.loss(*args, **kw)

