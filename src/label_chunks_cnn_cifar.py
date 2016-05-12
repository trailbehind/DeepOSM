""" 
    simple 1 layer network
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

  network = tflearn.input_data(shape=[None, image_size, image_size, on_band_count])
  network = conv_2d(network, 256, 16, activation='relu')
  network = max_pool_2d(network, 3)  
  #network = tflearn.fully_connected(network, 1024, activation='relu')
  softmax = tflearn.fully_connected(network, 2, activation='softmax')

  momentum = tflearn.optimizers.Momentum (learning_rate=.005,
                                          momentum=0.9, 
                                          lr_decay=0.0002, 
                                          name='Momentum')

  net = tflearn.regression(softmax, optimizer=momentum,
                           loss='categorical_crossentropy')

  # each epoch is 170 steps I think
  model = tflearn.DNN(net, tensorboard_verbose=0)
  model.fit(train_images, train_labels, n_epoch=50, shuffle=False, validation_set=(test_images, test_labels),
            show_metric=True, run_id='mlp')

  return model.predict(test_images)
