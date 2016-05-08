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

def train_neural_net(convolution_patch_size,
	                   bands_to_use,
	                   image_size,
	                   train_images, 
                     train_labels, 
                     test_images, 
                     test_labels,
                     number_of_batches,
                     batch_size):  

  on_band_count = 0
  for b in bands_to_use:
    if b == 1:
      on_band_count += 1

  train_images = train_images.astype(numpy.float32)
  train_images = (train_images - 127.5) / 127.5
    
  test_images = test_images.astype(numpy.float32)
  test_images = (test_images - 127.5) / 127.5

  # Convolutional network building
  network = input_data(shape=[None, image_size, image_size, on_band_count])
  network = conv_2d(network, 32, convolution_patch_size, activation='relu')
  network = max_pool_2d(network, 2)
  network = conv_2d(network, 64, convolution_patch_size, activation='relu')
  network = conv_2d(network, 64, convolution_patch_size, activation='relu')
  network = max_pool_2d(network, 2)
  network = fully_connected(network, 512, activation='relu')
  network = dropout(network, 0.5)
  network = fully_connected(network, 2, activation='softmax')
  network = regression(network, optimizer='adam',
                       loss='categorical_crossentropy',
                       learning_rate=0.001)

  # batch_size was originally 96
  # n_epoch was originally 50
  # each epoch is 170 steps I think
  # Train using classifier
  model = tflearn.DNN(network, tensorboard_verbose=0)
  model.fit(train_images, train_labels, n_epoch=int(number_of_batches/100), shuffle=False, validation_set=(test_images, test_labels),
            show_metric=True, batch_size=batch_size, run_id='cifar10_cnn')

  return model.predict(test_images)
