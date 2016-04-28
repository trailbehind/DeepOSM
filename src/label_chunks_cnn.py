'''
    train a deep convolutional neural network (CNN) with the OSM ways as labeled data for the imagery
    based on the 2nd tensroflow tutorial:
     https://github.com/trailbehind/DeepOSM/blob/246a2312828ccc0e5b395f8033825a46025232cc/mnist_tutorials/tutorial-mnist-expert.py
'''

import sys, time
from DataSet import DataSet, DataSets
import tensorflow as tf
import tensorflow.python.platform
import numpy


def train_neural_net(bands_to_use, 
                     image_size, 
                     train_images, 
                     train_labels, 
                     test_images, 
                     test_labels, 
                     convolution_patch_size,
                     number_of_batches,
                     batch_size):  

  on_band_count = 0
  for b in bands_to_use:
    if b == 1:
      on_band_count += 1

  data_sets = DataSets()
  data_sets.train = DataSet(on_band_count, train_images, train_labels, dtype=tf.float32)
  data_sets.test = DataSet(on_band_count, test_images, test_labels, dtype=tf.float32)
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

  # placeholder for inputs
  x = tf.placeholder("float", shape=[None, image_size*image_size*on_band_count])

  y_ = tf.placeholder(tf.float32, [None, 2])

  patch_size = convolution_patch_size

  # first layer of convolution
  W_conv1 = weight_variable([patch_size, patch_size, 1, 32])
  b_conv1 = bias_variable([32])

  x_image = tf.reshape(x, [-1,image_size,image_size,1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  # second layer
  W_conv2 = weight_variable([patch_size, patch_size, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([image_size/4 * image_size/4 * 64 * on_band_count, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, image_size/4*image_size/4*64 * on_band_count])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  #keep_prob = tf.placeholder("float")
  #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 2])
  b_fc2 = bias_variable([2])

  y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  sess.run(tf.initialize_all_variables())

  loss_total = 0

  print("TRAINING...") 
  t0 = time.time()
  for i in range(number_of_batches):
    batch = data_sets.train.next_batch(batch_size)
    # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    # print("step %d, training accuracy %g"%(i, train_accuracy))

    _, loss_val = sess.run([train_step, cross_entropy],
                           feed_dict={x: batch[0], y_: batch[1]})
    loss_total += loss_val
    print('step {}, loss = {}, loss rolling avg = {} '.format(i, loss_val, loss_total/(i+1)))

    # print the prediction matrix at this step
    # print "{} test labels are predicted to be ON".format(tf.argmax(y_conv,1).eval(feed_dict={x: data_sets.test.images}, session=sess).sum()/float(len(data_sets.test.images)))

  print("training time {0:.1f}s".format(time.time()-t0))
  print("test accuracy %g"%accuracy.eval(feed_dict={
      x: data_sets.test.images, y_: data_sets.test.labels,}))

  prediction=tf.argmax(y_conv,1)
  index = 0

  '''
  print prediction.eval(feed_dict={x: data_sets.test.images}, session=sess)  
  for pred in prediction.eval(feed_dict={x: data_sets.test.images}, session=sess):
    if pred == 1:
      print index
  
    index += 1
  '''
  return prediction.eval(feed_dict={x: data_sets.test.images}, session=sess)
