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

### Installing

It is first required to download/train the different datasets and models that are going to be attacked.

For MNIST and CIFAR it is necessary to run the following comand
```
python Setups/Data_and_Model/train_CIFAR_MNIST_models.py
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
and to install the inception-v3 net 
```
python Setups/Data_and_Model/setup_inception.py
```


For Inception Adversarially Trained Nets run

```
CHECKPOINT_DIR=./Models
mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvf adv_inception_v3_2017_08_18.tar.gz
mv adv_inception_v3.ckpt* ${CHECKPOINT_DIR}
rm adv_inception_v3_2017_08_18.tar.gz
```


End with an example of getting some data out of the system or using it for a little demo

## Running the tests

To run the experiments suggested in the manuscript, it is necessary to run separatly the following codes for each dataset.

These datasets are treated separately because of how the net net is stored and how each net then requires a slightly different implementation.

### MNIST

The normally trained net is attacked with:

```
# COMBI 
python Setups/MNIST_CIFAR_COMBI.py --epsilon=0.15 --dim_image=28 --num_channels=1 --dataset=mnist
# GENE
python Setups/MNIST_CIFAR_GENE.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=False
# SQUARE
python Setups/MNIST_CIFAR_SQUARE.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=False
# BOBYQA
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=False
```

The Adversarially trained net instead with 
```
# COMBI
python Setups/MNIST_CIFAR_COMBI.py --epsilon=0.15 --dim_image=28 --num_channels=1 --dataset=mnist --Adversary_trained=True
# GENE
python Setups/MNIST_CIFAR_GENE.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=True
# SQUARE
python Setups/MNIST_CIFAR_SQUARE.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=True
# BOBYQA 
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=mnist --Adversary_trained=True
```

### CIFAR

The normally trained net is attacked with

```
# COMBI
python Setups/MNIST_CIFAR_COMBI.py --epsilon=0.15 --dim_image=32 --num_channels=3 --dataset=cifar10
# GENE
python Setups/MNIST_CIFAR_GENE.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=False
# SQUARE
python Setups/MNIST_CIFAR_SQUARE.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=False
# BOBYQA
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=False
```

While the adversarially trained net was attacked with:

```
# COMBI
python Setups/MNIST_CIFAR_COMBI.py --epsilon=0.15 --dim_image=32 --num_channels=3 --dataset=cifar10 --Adversary_trained=True
# GENE
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=True
# SQUARE
python Setups/MNIST_CIFAR_SQUARE.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=False
# BOBYQA
python Setups/MNIST_CIFAR_BOBYQA.py --eps=0.15 --test_size=10 --dataset=cifar10 --Adversary_trained=True
```

### ImageNet

In the normal case   

```
# COMBI
python Setups/Inception_COMB_normal.py --sample_size=1 --epsilon=0.1 --max_queries=1500
# GENE
python Setups/Inception_GENE_normal.py --eps=0.1 --test_size=1 --max_queries=15000
# SQUARE
python Setups/Inception_SQUA_normal.py --test_size=1 --eps=0.1 --max_steps=15000
# BOBYQA
python Setups/Inception_BOBY_normal.py --eps=0.1 --max_queries=15000 --test_size=1
```

While in the adversary case

```
# COMBI
python Setups/Inception_COMB_adv.py --sample_size=1 --epsilon=0.1 --max_queries=15000
# BOBYQA
python Setups/Inception_BOBY_adv.py --eps=0.1 --test_size=1 --max_eval=15000
```

Note, that the BOBYQA and COMBI normal case expecially need a lot of memory. However on a desktop with 15GB of RAM there were no problems.

## Analysis of the Results

With the functions in ./Analysis_Results it is possible to generate the graphs in Figure 1 and 9 of the manuscript once the data are generated.

## Insight into BOBYQA

We suggest to check the normal attack to inception v3 if somone want to check in more detail the implementation of the attack as this is the most cleaned implementation
