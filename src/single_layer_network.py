'''
    simple 1 layer network
'''
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import numpy

def train(bands_to_use,
          image_size,
          train_images, 
          train_labels, 
          test_images, 
          test_labels,
          number_of_epochs,
          layer_type='one_layer_relu'):  
  '''
      trains a single layer neural network
      returns predicted values for the test_images
  '''
  on_band_count = 0
  for b in bands_to_use:
    if b == 1:
      on_band_count += 1

  # normalize 0-255 values to 0-1
  train_images = train_images.astype(numpy.float32)
  train_images = numpy.multiply(train_images, 1.0 / 255.0)
  test_images = test_images.astype(numpy.float32)
  test_images = numpy.multiply(test_images, 1.0 / 255.0)

  network = tflearn.input_data(shape=[None, image_size, image_size, on_band_count])
  if layer_type == 'one_layer_relu':
    network = tflearn.fully_connected(network, 2048, activation='relu')
  elif layer_type == 'one_layer_relu_conv':
    network = conv_2d(network, 256, 16, activation='relu')
    network = max_pool_2d(network, 3)  
  else:
    print("ERROR: exiting, unknown layer type for neural net")

  # classify as road or not road
  softmax = tflearn.fully_connected(network, 2, activation='softmax')

  # based on parameters from https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf
  momentum = tflearn.optimizers.Momentum (learning_rate=.005,
                                          momentum=0.9, 
                                          lr_decay=0.0002, 
                                          name='Momentum')

  net = tflearn.regression(softmax, optimizer=momentum, loss='categorical_crossentropy')

  model = tflearn.DNN(net, tensorboard_verbose=0)
  model.fit(train_images, train_labels, n_epoch=number_of_epochs, shuffle=False, validation_set=(test_images, test_labels),
            show_metric=True, run_id='mlp')
  
  # batch predictions on the test image set, to avoid a memory spike
  all_predictions = []
  for x in range(0, len(test_images)-100, 100):
    for p in model.predict(test_images[x:x+100]):
      all_predictions.append(p)
  remainder = len(test_images)-len(all_predictions)
  for p in model.predict(test_images[len(all_predictions):]):
      all_predictions.append(p)
  assert len(all_predictions) == len(test_images)

  return all_predictions
