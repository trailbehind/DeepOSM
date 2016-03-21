'''
    train a deep learning net with the OSM ways as labeled data for the imagery
    based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py
'''

import tensorflow as tf

class DataSet(object):
    '''
       a DataSet that Tensorflow likes to consume
    '''
    def __init__(self, images, labels, dtype=tf.float32):
        """
        Construct a DataSet.
        `dtype` can be either `uint8` to leave the input as `[0, 255]`,
        or `float32` to rescale into `[0, 1]`.
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        assert images.shape[0] == labels.shape[0], (
                            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1

        # Store the width and height of the images before flattening it, if only for reference.
        image_height, image_width = images.shape[1], images.shape[2]
        self.original_image_width = image_width
        self.original_image_height = image_height

        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0]
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
      """Return the next `batch_size` examples from this data set."""
      start = self._index_in_epoch
      self._index_in_epoch += batch_size
      if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = numpy.arange(self._num_examples)
          numpy.random.shuffle(perm)
          self._images = self._images[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

class DataSets(object):
    pass

