# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-10 dataset classification task.
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import numpy

def train_neural_net(train_images, 
                     train_labels, 
                     test_images, 
                     test_labels):  

  train_images = train_images.astype(numpy.float32)
  train_images = (train_images - 127.5) / 127.5
    
  test_images = test_images.astype(numpy.float32)
  test_images = (test_images - 127.5) / 127.5

  # Convolutional network building
  network = input_data(shape=[None, 64, 64, 4])
  network = conv_2d(network, 32, 3, activation='relu')
  #network = max_pool_2d(network, 2)
  network = conv_2d(network, 64, 3, activation='relu')
  network = conv_2d(network, 64, 3, activation='relu')
  network = max_pool_2d(network, 2)
  network = fully_connected(network, 512, activation='relu')
  #network = dropout(network, 0.5)
  network = fully_connected(network, 2, activation='softmax')
  network = regression(network, optimizer='adam',
                       loss='categorical_crossentropy',
                       learning_rate=0.001)

  # Train using classifier
  model = tflearn.DNN(network, tensorboard_verbose=0)
  model.fit(train_images, train_labels, n_epoch=50, shuffle=True, validation_set=(test_images, test_labels),
            show_metric=True, batch_size=96, run_id='cifar10_cnn')
