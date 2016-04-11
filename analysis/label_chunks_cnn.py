'''
    train a deep convolutional neural network (CNN) with the OSM ways as labeled data for the imagery
    based on the 2nd tensroflow tutorial:
     https://github.com/trailbehind/Deep-OSM/blob/246a2312828ccc0e5b395f8033825a46025232cc/mnist_tutorials/tutorial-mnist-expert.py
'''

import sys
from DataSet import DataSet, DataSets
import tensorflow as tf
import tensorflow.python.platform
import numpy


def train_neural_net(train_images, train_labels, test_images, test_labels):  
  data_sets = DataSets()
  data_sets.train = DataSet(train_images, train_labels, dtype=tf.uint8)
  data_sets.test = DataSet(test_images, test_labels, dtype=tf.uint8)
  print("CREATED DATASET: {} training images, {} test images, with {} training labels, and {} test labels".format(len(train_images), len(test_images), len(train_labels), len(test_labels)))

  sess = tf.InteractiveSession()

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  image_size = 256
  # placeholder for inputs
  x = tf.placeholder("float", shape=[None, image_size*image_size])

  y_ = tf.placeholder(tf.float32, [None, 2])

  # first layer of convolution
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  x_image = tf.reshape(x, [-1,image_size,image_size,1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  # second layer

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([64 * 64 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 64*64*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 2])
  b_fc2 = bias_variable([2])

  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  sess.run(tf.initialize_all_variables())

  batch_size = 100
  for i in range(int(len(train_images)/batch_size)):
    batch = data_sets.train.next_batch(batch_size)
    #if i%5 == 0:
    train_accuracy = accuracy.eval(feed_dict={
      x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print("test accuracy %g"%accuracy.eval(feed_dict={
      x: data_sets.test.images, y_: data_sets.test.labels, keep_prob: 1.0}))
 

if __name__ == '__main__':
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
    from marshall_osm_data import OSMDataNormalizer
    odn = OSMDataNormalizer()
    
    # process into matrices
    odn.process_geojson()
    odn.process_rasters()
    # create a DataSet that Tensorflow likes
    train_neural_net(odn.train_images, odn.train_labels, odn.test_images, odn.test_labels)
  else:
    print(parameters_message)