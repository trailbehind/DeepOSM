'''
    train a softmax regression with the OSM ways as labeled data for the imagery
    based on the 1st tensroflow tutorial https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
'''

import sys
from marshall_osm_data import OSMDataNormalizer
from DataSet import DataSet, DataSets
import tensorflow as tf
import tensorflow.python.platform
import numpy


def train_neural_net():  
  odn = OSMDataNormalizer()
  
  # process into matrices
  odn.process_geojson()
  odn.process_rasters()

  # create a DataSet that Tensorflow likes
  data_sets = DataSets()
  data_sets.train = DataSet(odn.train_images, odn.train_labels, dtype=tf.uint8)
  data_sets.test = DataSet(odn.test_images, odn.test_labels, dtype=tf.uint8)
  print("CREATED DATASET: {} training images, {} test images, with {} training labels, and {} test labels".format(len(odn.train_images), len(odn.test_images), len(odn.train_labels), len(odn.test_labels)))

  # run a TensorFlow session
  sess = tf.InteractiveSession()
  # Create the model
  x = tf.placeholder(tf.float32, [None, odn.thumb_size*odn.thumb_size])
  W = tf.Variable(tf.zeros([odn.thumb_size*odn.thumb_size, 2]))
  b = tf.Variable(tf.zeros([2]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  # Train
  tf.initialize_all_variables().run()
  batch_size = 10
  for i in range(batch_size):
    batch_xs, batch_ys = data_sets.train.next_batch(len(odn.train_images)/batch_size)
    train_step.run({x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(accuracy.eval({x: data_sets.test.images, y_: data_sets.test.labels}))


parameters_message = "parameters are: download-data, train"
if len(sys.argv) == 1:
  print(parameters_message)
elif sys.argv[1] == 'download-data':
  if len(sys.argv) < 3:
    print('download-data requires a Mapzen APIkey as the 2nd parameter')
  else:
    odn = OSMDataNormalizer(sys.argv[2])
    odn.download_tiles()
elif sys.argv[1] == 'train':
  train_neural_net()
else:
  print(parameters_message)
