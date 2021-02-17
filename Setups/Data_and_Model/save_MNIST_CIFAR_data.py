## train_models.py -- train the neural network models for attacking
##
## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_2 = os.path.dirname(dir_path)
main_dir = os.path.abspath(os.path.join(dir_path_2, os.pardir))

if not os.path.isdir('Models'):
    os.makedirs('Models')

if not os.path.isdir('Data'):
    os.makedirs('Data')


if not os.path.isdir('Data/CIFAR-10'):
    os.makedirs('Data/CIFAR-10')

if not os.path.exists(main_dir + "/Data/CIFAR-10/cifar-10-batches-bin"):
    print('Downloading the Dataset')
    urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                main_dir + '/Data/CIFAR-10/cifar-data.tar.gz')
    command = "tar -xzvf " + main_dir + "/Data/CIFAR-10/cifar-data.tar.gz -C "+ main_dir + "/Data/CIFAR-10"
    os.popen(command).read()
