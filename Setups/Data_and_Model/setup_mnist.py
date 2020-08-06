## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
main_dir = os.path.abspath(os.path.join(main_dir, os.pardir))

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self):
        if not os.path.exists(main_dir + "/Data/MNIST/data"):
            os.mkdir(main_dir + "/Data/MNIST")
            os.mkdir(main_dir + "/Data/MNIST/data")
        if not os.path.exists(main_dir + "/Data/MNIST/data/train-images-idx3-ubyte.gz"):
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:
                print('retrieving the data')
                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, main_dir + "/Data/MNIST/data/"+name)

        train_data = extract_data(main_dir + "/Data/MNIST/data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels(main_dir + "/Data/MNIST/data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data(main_dir + "/Data/MNIST/data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels(main_dir + "/Data/MNIST/data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


