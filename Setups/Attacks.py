"""
This Script allows to run attacks to nets trained on either ImageNet or Cifar10 with or without the
adversary defence by MadryLab. 
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append("./")
import tensorflow as tf
import torchvision.models as models_ch
import torch as ch
import numpy as np
import random
import time

import pickle

from Setups.Data_and_Model.setup_cifar import CIFAR#, CIFARModel
from Setups.Data_and_Model.setup_mnist import MNIST#, MNISTModel
from Setups.Data_and_Model.setup_inception_2 import ImageNet
from Setups.Data_and_Model.wrapper_model_loss_f import wrapper_model, wrapper_loss

from Attack_Code.BOBYQA.BOBYQA_Attack_Adversary_channels_2 import BlackBox_BOBYQA
from Attack_Code.BOBYQA.BOBYQA_Attack_random_direction import BlackBox_BOBYQA_random_proj
from Attack_Code.Combinatorial.attacks.parsimonious_attack_madry import ParsimoniousAttack
from Attack_Code.Square_Attack.attack_madry import square_attack_linf
from Attack_Code.GenAttack.genattack_tf2_PyTorch import GenAttack2
from Attack_Code.Frank_Wolfe.FW_black import FW_black

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

from PIL import Image
# import torch as ch
from robustness.datasets import CIFAR as CIFAR_robustness
from robustness.datasets import ImageNet as Imagenet_robustness
from robustness.model_utils import make_and_restore_model

flags = tf.app.flags
flags.DEFINE_string('dataset', 'cifar10', 'model name')
flags.DEFINE_string('attack', 'boby', 'Attack considered')
flags.DEFINE_integer('test_size', 1, 'Number of test images.')
flags.DEFINE_integer('max_evals', None, 'Maximum number of function evaluations.')
flags.DEFINE_integer('print_every', 10, 'Every iterations the attack function has to print out.')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_bool('Adversary_trained', True, ' Use the adversarially trained nets')
flags.DEFINE_string('description', '', 'Further description describing the results')
flags.DEFINE_float('eps', None, 'perturbation energy')
flags.DEFINE_integer('batch_size', None, 'Dimension of the optimisation domain.')
flags.DEFINE_bool('subspace_attack', False, ' Attack only a fixed number of pixels with highest variability')
flags.DEFINE_integer('subspace_dimension', None, 'Dimension of the subspace optimisation domain when doing subspace attack.')

# BOPBYQA parameters
flags.DEFINE_string('interpolation', 'block', 'Interpolation inbetween grod elements in the BOBYQA attack')
flags.DEFINE_bool('use_resize', True, 'if using hierarchical approach')
flags.DEFINE_integer('n_channels', 3, 'n channels in the perturbation grid')
flags.DEFINE_bool('save', True, 'If saving the results')
flags.DEFINE_float('max_f', 1.3 , 'Maximum number of function evaluations in the BOBYQA attack')
flags.DEFINE_string('over', 'over', 'Kind of interpolation within block in the BOBYQA attack')
flags.DEFINE_bool('rounding', True, 'If to include the rounding possibility in the attacks')

# COMBI parameters
flags.DEFINE_string('asset_dir', default=main_dir+'/Attack_Code/Combinatorial/assets', help='directory assets')
flags.DEFINE_bool('targeted', default=True, help='bool on targeted')
flags.DEFINE_integer('max_iters', 1, help='maximum iterations') 
flags.DEFINE_integer('block_size', default=128, help='blck size') 
flags.DEFINE_bool('no_hier', default=False, help='bool on hierarchical attack') #########
flags.DEFINE_integer('dim_image', default=32, help='Dimension of the image that we feed as an input')
flags.DEFINE_integer('num_channels', default=3, help='Channels of the image that we feed as an input')

# SQUARE parameters
flags.DEFINE_float('p_init', default=None , help='dimension of the blocks')

# Gene parameters
flags.DEFINE_float('mutation_rate', default=0.005, help='Mutation rate')
flags.DEFINE_float('alpha', default=0.20, help='Step size')
flags.DEFINE_integer('pop_size', default=6, help='Population size')
flags.DEFINE_integer('resize_dim', default=96, help='Reduced dimension for dimensionality reduction')
flags.DEFINE_bool('adaptive', default=True, help='Turns on the dynamic scaling of mutation prameters')

#FW parameters
flags.DEFINE_integer('att_iter', default=10000, help='Attack_Iterations')
flags.DEFINE_integer('grad_est_batch_size', default=25, help='Dimension of batch for gradient estimation')
flags.DEFINE_float('l_r', default=None, help='Learning Rate')
flags.DEFINE_float('delta', default=0.01, help='radius on which gradient is learnt')
flags.DEFINE_float('beta1', default=None, help='Momentum Parameters')
flags.DEFINE_string('sensing_type', default='gaussian', help='sensing type')

FLAGS = flags.FLAGS

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def generate_data(data, samples, dataset, targeted=True, start=0):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    """
    inputs = []
    targets = []
    labels = []
    for i in range(samples):
        if dataset == 'cifar10':
            seq = range(data.test_labels.shape[1])
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                labels.append(data.test_labels[start+i])
        elif dataset == 'ImageNet':
            num_labels = data.test_labels.shape[1]
            inputs.append(data.test_data[i])
            labels.append(data.test_labels[i])
            other_labels = [x for x in range(num_labels) if data.test_labels[i][x] == 0]
            random_target = [0 for _ in range(num_labels)]
            random_target[np.random.choice(other_labels)] = 1
            targets.append(random_target)

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)

    return inputs, targets, labels


if __name__ == '__main__':
    # Set Parameters of the attacks

    if ch.cuda.is_available():
        free_GPU_idx=get_freer_gpu()
        ch.cuda.set_device(free_GPU_idx)
        print('[INFO] using cuda device ', free_GPU_idx)
    with ch.no_grad():
        if FLAGS.max_evals is None:
            if (FLAGS.dataset == 'mnist' or 
                    FLAGS.dataset == 'cifar10'):
                FLAGS.max_evals = 3000
            elif FLAGS.dataset == 'ImageNet':
                FLAGS.max_evals = 15000
        
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        # Initialise list to save the results
        list_attack = []
        single_output = False # Need this to deal wiht the different net for non adversary
                            # trained net with imagenet
        # load network
        if FLAGS.dataset == "mnist":
            if FLAGS.Adversary_trained:
                data, model = MNIST(), MNISTModel(main_dir + "/Models/mnist-distilled-100", sess, use_log)
            else:
                data, model = MNIST(), MNISTModel(main_dir + "/Models/mnist", sess, use_log)
            
            if FLAGS.eps is None:
                epsilons = []
            else:
                epsilons = FLAGS.eps

        elif FLAGS.dataset == "cifar10":
            ds = CIFAR_robustness('./Data/CIFAR-10')
            data = CIFAR()
            if FLAGS.Adversary_trained:
                model,_ = make_and_restore_model(arch='resnet50', dataset=ds, 
                                        resume_path='./Models/cifar_robust.pt',
                                        parallel=False)
            else:
                model,_ = make_and_restore_model(arch='resnet50', dataset=ds, 
                                        resume_path='./Models/cifar_nat.pt',
                                        parallel=False)
                
            if FLAGS.eps is None:
                epsilons = [0.01, 0.005]
            else:
                epsilons = [FLAGS.eps]

        elif FLAGS.dataset == "ImageNet":
            ds = Imagenet_robustness('./Data/ImagNet/images')
            data = ImageNet('', dimension=299)
            if FLAGS.Adversary_trained:
                model,_ = make_and_restore_model(arch='resnet50', dataset=ds, 
                                        resume_path='./Models/imagenet_linf_8.pt',
                                        parallel=False)
            else:
                model = models_ch.resnet50(pretrained=True)
                single_output=True
            
            if FLAGS.eps is None:
                if not FLAGS.Adversary_trained:
                    epsilons = [0.01, 0.1]
                else:
                    epsilons = [0.02]
            else:
                epsilons = [FLAGS.eps]
        
        if ch.cuda.is_available():
            model.eval()
            model.to('cuda')
            print('[INFO] Using Cuda')
            model = wrapper_model(model.float(), FLAGS.attack, single_output, cuda=True)
        else:
            model.eval()
            ch.set_num_threads(8)
            model = wrapper_model(model.float(), FLAGS.attack, single_output)
        
        
        for eps in epsilons:
            list_attack = []

            FLAGS.eps=eps
            
            # loading the data
            all_inputs, all_targets, all_labels = generate_data(data, dataset=FLAGS.dataset,
                                        samples=FLAGS.test_size, targeted=True,
                                        start=0)

            # Set Loading and Saving directories
            if FLAGS.dataset == 'mnist':
                saving_dir = './Results/MNIST/'
            elif FLAGS.dataset == 'cifar10':
                saving_dir = './Results/CIFAR/'
            elif FLAGS.dataset == 'ImageNet':
                saving_dir = './Results/Imagenet/'
            
            if FLAGS.attack == 'boby':
                if FLAGS.batch_size is None:
                    FLAGS.batch_size = 20
                saving_name = (saving_dir+FLAGS.attack +'_adversary_' + str(FLAGS.Adversary_trained) +
                                    '_interpolation_' + FLAGS.interpolation +
                                    '_eps_'+str(FLAGS.eps) +
                                    '_max_eval_'+ str(FLAGS.max_evals) + 
                                    '_n_channels_' + str(FLAGS.n_channels) + 
                                    '_over_' + str(FLAGS.over) + 
                                    '_max_f_' + str(FLAGS.max_f) + 
                                    '_rounding_' + str(FLAGS.rounding) + 
                                    '_subspace_attack_' + str(FLAGS.subspace_attack) +
                                    '_subspace_dimension_' + str(FLAGS.subspace_dimension) +
                                    FLAGS.description + '.txt')  
            elif FLAGS.attack == 'combi':
                if FLAGS.batch_size is None:
                    FLAGS.batch_size = 64
                saving_name = (saving_dir+FLAGS.attack +'_adversary_' + str(FLAGS.Adversary_trained) +
                                    '_eps_'+str(FLAGS.eps) +
                                    '_max_eval_'+ str(FLAGS.max_evals) + 
                                    '_max_iters_' + str(FLAGS.max_iters) + 
                                    '_block_size_' + str(FLAGS.block_size) +
                                    '_batch_size_' + str(FLAGS.batch_size) +
                                    '_no_hier_' + str(FLAGS.no_hier) + 
                                    '_subspace_attack_' + str(FLAGS.subspace_attack) +
                                    '_subspace_dimension_' + str(FLAGS.subspace_dimension) +
                                    FLAGS.description + '.txt')  
            elif FLAGS.attack == 'square':
                if FLAGS.p_init is None:
                    if FLAGS.dataset == 'ImageNet':
                        FLAGS.p_init = 0.01
                    else:
                        FLAGS.p_init = 0.1

                saving_name = (saving_dir+FLAGS.attack +'_adversary_' + str(FLAGS.Adversary_trained) +
                                    '_eps_'+str(FLAGS.eps) +
                                    '_max_eval_'+ str(FLAGS.max_evals) + 
                                    '_p_init_' + str(FLAGS.p_init) +
                                    '_subspace_attack_' + str(FLAGS.subspace_attack) +
                                    '_subspace_dimension_' + str(FLAGS.subspace_dimension) +
                                    FLAGS.description + '.txt')  
            elif FLAGS.attack == 'gene':
                saving_name = (saving_dir+FLAGS.attack +'_adversary_' + str(FLAGS.Adversary_trained) +
                                    '_eps_'+str(FLAGS.eps) +
                                    '_max_eval_'+ str(FLAGS.max_evals) + 
                                    '_pop_size_'+ str(FLAGS.pop_size) +
                                    '_mutation_rate_'+ str(FLAGS.mutation_rate) +
                                    '_alpha_'+ str(FLAGS.alpha) +
                                    '_resize_dim_'+ str(FLAGS.resize_dim) +
                                    '_adaptive_'+ str(FLAGS.adaptive) +
                                    '_subspace_attack_' + str(FLAGS.subspace_attack) +
                                    '_subspace_dimension_' + str(FLAGS.subspace_dimension) +
                                    FLAGS.description + '.txt')  
            elif FLAGS.attack == 'FW':
                if FLAGS.l_r is None:
                    FLAGS.l_r = 0.8
                if FLAGS.beta1 is None:
                    if FLAGS.attack == 'ImageNet':
                        FLAGS.beta1 = 0.999
                    else:
                        FLAGS.beta1 = 0.99
                
                saving_name = (saving_dir+FLAGS.attack +'_adversary_' + str(FLAGS.Adversary_trained) +
                                    '_eps_'+str(FLAGS.eps) +
                                    '_max_eval_'+ str(FLAGS.max_evals) + 
                                    '_att_iter_'+ str(FLAGS.att_iter) +
                                    '_grad_est_batch_size_' + str(FLAGS.grad_est_batch_size) +
                                    '_l_r_'+str(FLAGS.l_r)+
                                    '_delta_'+str(FLAGS.delta)+
                                    '_beat1_'+str(FLAGS.beta1)+
                                    '_sensing_type_'+str(FLAGS.sensing_type)+
                                    '_subspace_attack_' + str(FLAGS.subspace_attack) +
                                    '_subspace_dimension_' + str(FLAGS.subspace_dimension) +
                                    FLAGS.description + '.txt')  
            
            # Loading the previous results obtained with the same saving directory
            already_done = 0
            # if we are not saving, i.e. doing a test, we want to start from the same image
            if FLAGS.save: 
                if os.path.exists(saving_name):
                    if os.path.getsize(saving_name)>0:
                        with open(saving_name, "rb") as fp:
                            list_attack = pickle.load(fp)
                        already_done = len(list_attack)
            
                if already_done>0:
                    print(already_done,'attacks')
                    found = False
                    # set the label and target of the image that has been last attacked
                    lab = list_attack[-1][2]
                    tar = list_attack[-1][3] 
                    # iterate throught the data to find at what point we are
                    while not found:
                        # print(all_labels[already_done-1:already_done],all_targets[already_done-1:already_done],
                        #      ' instead of ', lab, tar)
                        try:
                            lab_ = np.argmax(all_labels[already_done-1:already_done])
                            tar_ = np.argmax(all_targets[already_done-1:already_done])
                        except:
                            already_done = 0

                        already_done +=1
                        if lab == lab_ and tar==tar_:
                            found = True
                            already_done -= 1
                        

                        if already_done==398:
                            print('ERROR LOADING')
                            found=True
                            break

            
            print('[EXPERIMENTAL SETUP] Attacking', FLAGS.test_size, 'test images with ', FLAGS.eps,' energy')
            print('                    ', already_done, ' attacks have already been conducted.')
            
            for i in range(already_done,FLAGS.test_size):
                inputs = all_inputs[i:i+1]
                if FLAGS.attack == 'combi':
                    inputs = inputs[0]
                targets = all_targets[i:i+1]
                labels = all_labels[i:i+1]        

                print('[L1] Image of class ',np.argmax(labels),' targeted to ', np.argmax(targets),'.')

                original_predict = model.predict(inputs)
                original_predict = np.squeeze(original_predict)
                original_prob = np.sort(original_predict)
                original_class = np.argsort(original_predict)
                if original_class[-1] != np.argmax(labels):
                    print("skip wrongly classified image no. {}, original class {}, classified as {}".format(
                                        i, np.argmax(labels), original_class[-1]))
                    continue

                loss_func = wrapper_loss(FLAGS.attack, targets, model)
                if FLAGS.attack=='boby':
                    attack = BlackBox_BOBYQA(loss_func, batch_size=FLAGS.batch_size ,
                                            interpolation = FLAGS.interpolation,
                                            n_channels_input=FLAGS.n_channels,
                                            print_every=FLAGS.print_every, use_resize=FLAGS.use_resize, 
                                            eps=FLAGS.eps, max_eval=FLAGS.max_evals,
                                            over=FLAGS.over, rounding=FLAGS.rounding,
                                            max_f=FLAGS.max_f, subspace_attack=FLAGS.subspace_attack,
                                            subspace_dim=FLAGS.subspace_dimension)

                    result = attack.attack_batch(inputs, targets)
                if FLAGS.attack=='boby_random':
                    attack = BlackBox_BOBYQA_random_proj(loss_func, batch_size=50 ,
                                            n_channels_input=FLAGS.n_channels,
                                            print_every=FLAGS.print_every,
                                            eps=FLAGS.eps, max_eval=FLAGS.max_evals,
                                            max_f=FLAGS.max_f, subspace_attack=FLAGS.subspace_attack,
                                            subspace_dim=FLAGS.subspace_dimension, delta=FLAGS.delta)

                    result = attack.attack_batch(inputs, targets)
                elif FLAGS.attack=='combi':
                    result = ParsimoniousAttack(loss_func, inputs, 
                                                np.argmax(targets[0]), FLAGS)
                elif FLAGS.attack=='square':
                    result = square_attack_linf(model=model, x=inputs, 
                                            y=np.array(targets[0], dtype=int), 
                                            eps = FLAGS.eps, n_iters=FLAGS.max_evals, 
                                            p_init=FLAGS.p_init, targeted=True, 
                                            loss_type='cross_entropy', 
                                            print_every=FLAGS.print_every,subspace_attack=FLAGS.subspace_attack,
                                            subspace_dim=FLAGS.subspace_dimension)
                elif FLAGS.attack=='gene':
                    attack = GenAttack2(model=model, pop_size=FLAGS.pop_size, mutation_rate=FLAGS.mutation_rate,
                                        eps=FLAGS.eps, max_evals=FLAGS.max_evals , alpha=FLAGS.alpha,
                                        resize_dim=FLAGS.resize_dim, adaptive=FLAGS.adaptive, 
                                        num_classes=len(targets[0]), input_dim=inputs.shape[1])
                    result = attack.attack(inputs, np.argmax(targets[0]))
                elif FLAGS.attack=='FW':
                    attack = FW_black(loss_f=loss_func, att_iter=FLAGS.att_iter, grad_est_batch_size=FLAGS.grad_est_batch_size,
                        eps=FLAGS.eps, lr=FLAGS.l_r, delta=FLAGS.delta, sensing_type=FLAGS.sensing_type, 
                        q_limit=FLAGS.max_evals, beta1=FLAGS.beta1)
                    result = attack.attack(inputs, np.argmax(targets[0]))
                                            

                adv, eval_costs, summary, Success = result
                
                adversarial_predict = model.predict(adv)
                adversarial_predict = np.squeeze(adversarial_predict)
                adversarial_prob = np.sort(adversarial_predict)
                adversarial_class = np.argsort(adversarial_predict)

                list_attack.append([eval_costs, adversarial_predict,
                                np.argmax(labels), np.argmax(targets)])
                
                print("[STATS][L1] no={}, success: {}, prev_class = {}, new_class = {}".format(i, Success, original_class[-1], adversarial_class[-1]))
                sys.stdout.flush()

                if FLAGS.save:
                    with open(saving_name , "wb") as fp:
                            pickle.dump(list_attack, fp)
