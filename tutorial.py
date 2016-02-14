import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import tensorflow.python.platform

number_of_digits = 10
pixels_per_side = 28
sq_pixels = pixels_per_side*pixels_per_side

# create a placeholder for a tensor of floats of any length, with size sq_pixels (28*28=784)
x = tf.placeholder(tf.float32, [None, sq_pixels])

# initliaze weights and biases to zeroes since they are learned
W = tf.Variable(tf.zeros([sq_pixels, number_of_digits]))
b = tf.Variable(tf.zeros([number_of_digits]))

# this is the model for our predictions
y = tf.nn.softmax(tf.matmul(x, W) + b)

# create a placeholder for a tensor of floats of any length, with size number_of_digits
# this is for the real values in one-hot format
y_ = tf.placeholder(tf.float32, [None, number_of_digits])

# compute how inaccurate the model is
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# apply gradient descent optimization
# minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.01
learning_step = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_step).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})