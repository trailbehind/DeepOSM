""" 
    simple 1 convolution network
    inspired by CIFAR 10 classifier kind of: (https://www.cs.toronto.edu/~kriz/cifar.html)
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import numpy

def train_neural_net(bands_to_use,
	                   image_size,
	                   train_images, 
                     train_labels, 
                     test_images, 
                     test_labels):  
  on_band_count = 0
  for b in bands_to_use:
    if b == 1:
      on_band_count += 1

  train_images = train_images.astype(numpy.float32)
  train_images = numpy.multiply(train_images, 1.0 / 255.0)
    
  test_images = test_images.astype(numpy.float32)
  test_images = numpy.multiply(test_images, 1.0 / 255.0)

  input_layer = tflearn.input_data(shape=[None, image_size, image_size, on_band_count])
  dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                   regularizer='L2', weight_decay=0.001)
  dropout1 = tflearn.dropout(dense1, 0.5)
  dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                   regularizer='L2', weight_decay=0.001)
  dropout2 = tflearn.dropout(dense2, 0.5)
  softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

  # Regression using SGD with learning rate decay and Top-3 accuracy
  sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
  net = tflearn.regression(softmax, optimizer=sgd,
                           loss='categorical_crossentropy')

  # each epoch is 170 steps I think
  model = tflearn.DNN(net, tensorboard_verbose=0)
  model.fit(train_images, train_labels, n_epoch=60, shuffle=False, validation_set=(test_images, test_labels),
            show_metric=True, run_id='mlp')

  return model.predict(test_images)
