# An Empirical Study of DFO Algorithms for Targeted Black Box Attacks in DNNs

This repository contains the scripts that allow the reproduction of the comparative results in the manuscript "An Empirical Study of Derivative Free Optimization Algorithms for Targeted Black Box Attacks in Deep Neural Networks".


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

```
pip install -r requirements.txt 
```

Alternatively, one can directly uplaod one of the following conda environments according to the accessability to a GPU.
```
# With GPU
conda env create -f Adv_Attacks_GPU.yml 
# Without GPU 
conda env create -f Adv_Attacks_CPU.yml 
```
Note that if using the CPUs, the robustness package has to be modified to allow it to load with the cpu device. To do this, open the robustness/model_utils.py documnet and 
in the fucntion make_and_restore_model comment out the line 
```
model = model.cuda() 
```
and modify the line 
```
checkpoint = ch.load(resume_path, pickle_module=dill)
```
to
```
checkpoint = ch.load(resume_path, pickle_module=dill,map_location=ch.device('cpu'))
```

### Downloading the Datasets

For MNIST and CIFAR it is necessary to run the following comand
```
CHECKPOINT_DATA=./Data
mkdir ${CHECKPOINT_DATA}
python Setups/Data_and_Model/save_MNIST_CIFAR_data.py
```
To download the ImageNet dataset
```
CHECKPOINT_DIR=./Data/ImageNet
mkdir ${CHECKPOINT_DIR}
wget http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz
tar -xvf img.tar.gz -C ./Data/ImageNet
rm img.tar.gz
mv ./Data/ImageNet/imgs ./Data/ImageNet/images
rm ./Data/ImageNet/imgs

```

### Downloading the Neural Networks


We first have to make the directory for the mdoels via
```
CHECKPOINT_DIR=./Models
mkdir ${CHECKPOINT_DIR}
```

and then the adversarially trained nets for CIFAR and ImageNet are downloaded directly from [a link](https://github.com/MadryLab/robustness).

We considered the ε = 8/255 for the cifar10 case [a link](https://www.dropbox.com/s/c9qlt1lbdnu9tlo/cifar_linf_8.pt?dl=0) and also the ε = 8/255 in the ImageNet case [a link](https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0).

The nets for the non-adversarial case are inbuilt in Py-Torch.

## Setting the Parameters for the tests

There is a unique script whihc will run all of the different attacks that we might consider. This script has to be called with different flags that either specify the considered setup, or the parameters of the attack.



### Flags for Setup

- 'dataset' -- Defines the dataset on which the nets are trained and is either 'cifar10' or 'ImageNet'
- 'attack' -- Defines the kind of attack considered. For the BOBYQA choose 'boby', For the Parsimonious choose 'combi', for Square choose 'square', for Frank-Wolfe choose 'FW', and for GenAttack consider 'gene'.
- 'test_size' -- Number of test images.
- 'max_evals' -- 'Maximum number of function evaluations.
- 'print_every' -- number of iterations at which the attack function has to print out.
- 'seed' -- random seed.
- 'Adversary_trained' -- Use the adversarially trained nets.
- 'description' -- Further description to include while saving the results.
- 'eps' -- perturbation energy.
- 'batch_size' -- 'Dimension of the optimisation domain.
- 'subspace_attack' -- Attack only a fixed number of pixels with highest variability.
- 'subspace_dimension' -- Dimension of the subspace optimisation domain when doing subspace attack.
- 'save' -- If saving the results.

### BOPBYQA parameters
- 'interpolation' -- Interpolation inbetween grod elements in the BOBYQA attack. It is either 'block' or 'linear'. # This is not active as an option at the moment.
- 'use_resize' -- if using hierarchical approach.
- 'n_channels' -- n channels in the perturbation grid.
- 'max_f' -- Maximum number of function evaluations in the BOBYQA attack, $\kappa$ in the paper.
- 'over' -- Kind of interpolation within block in the BOBYQA attack. This is either 'over' or 'linaer'.
- 'rounding' -- If to include the rounding possibility in the attacks, i.e. checking that the the perturbation projected to the boundaries is better or not.

### COMBI parameters
- 'asset_dir' --directory assets.
- 'targeted' -- bool on targeted.
- 'max_iters' -- maximum iterations.
- 'block_size' -- blck size.
- 'no_hier' -- bool on hierarchical attack.
- 'dim_image' -- Dimension of the image that we feed as an input.
- 'num_channels' -- Channels of the image that we feed as an input.

### SQUARE parameters
- 'p_init' -- dimension of the blocks.

### Gene parameters
- 'mutation_rate' -- Mutation rate.
- 'alpha' -- Step size.
- 'pop_size' -- Population size.
- 'resize_dim' -- Reduced dimension for dimensionality reduction.
- 'adaptive' -- Turns on the dynamic scaling of mutation prameters.

### FW parameters
- 'att_iter' -- Attack_Iterations.
- 'grad_est_batch_size' -- Dimension of batch for gradient estimation.
- 'l_r' -- Learning Rate.
- 'delta' -- radius on which gradient is learnt.
- 'beta1' -- Momentum Parameters.
- 'sensing_type' -- sensing type.

## Example on how to run a test

The following is an example on how to run an attack on ImageNet with the BOBYQA method with the maximum perturbation energy being $\varepsilon=0.1$ and the net not being adversarially trained. Many of the parameters in each attack are given by default and there is no need to specify them.

```
python Setups/Attacks.py --dataset=ImageNet --attack=boby --eps=0.1 --test_size=1 --Adversary_trained=False
```


## Analysis of the Results

With the functions in ./Analysis_Results it is possible to generate the graphs in Figure 1 and 9 of the manuscript once the data are generated.

## Insight into BOBYQA

We suggest to check the normal attack to inception v3 if somone want to check in more detail the implementation of the attack as this is the most cleaned implementation
