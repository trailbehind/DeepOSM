"""A simple 1 layer network."""

from __future__ import division, print_function, absolute_import

import numpy
import pickle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d

from src.training_data import CACHE_PATH, load_training_tiles, equalize_data, \
    format_as_onehot_arrays, has_ways_in_center

MODEL_METADATA_PATH = 'model_metadata.pickle'


def train_on_cached_data(raster_data_paths, neural_net_type, bands, tile_size, number_of_epochs):
    """Load tiled/cached training data in batches, and train the neural net."""
    training_images = []
    onehot_training_labels = []
    model = None

    # the number of times to pull 10K images from disk, which produce about 200 training images
    # because we want half on, half off
    NUMBER_OF_BATCHES = 100

    # there are usually 100+ images with road through the middle, out of every 10,000
    EQUALIZATION_BATCH_SIZE = 10000

    for x in range(0, NUMBER_OF_BATCHES):
        new_label_paths = load_training_tiles(EQUALIZATION_BATCH_SIZE)
        print("Got batch of {} labels".format(len(new_label_paths)))
        new_training_images, new_onehot_training_labels = format_as_onehot_arrays(new_label_paths)
        equal_count_way_list, equal_count_tile_list = equalize_data(new_onehot_training_labels,
                                                                    new_training_images, False)
        [training_images.append(i) for i in equal_count_tile_list]
        [onehot_training_labels.append(l) for l in equal_count_way_list]

        # once we have 100 test_images, train on a mini batch
        if len(training_images) >= 100:
            # continue training the model with the new data set
            model = train_with_data(onehot_training_labels, training_images, neural_net_type, bands,
                                    tile_size, number_of_epochs, model)
            training_images = []
            onehot_training_labels = []

    save_model(model, neural_net_type, bands, tile_size)

    return model


def train_with_data(onehot_training_labels, training_images,
                    neural_net_type, band_list, tile_size, number_of_epochs, model):
    """Package data for tensorflow and analyze."""
    npy_training_labels = numpy.asarray(onehot_training_labels)

    # normalize 0-255 values to 0-1
    norm_training_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in training_images])
    norm_train_images = norm_training_images.astype(numpy.float32)
    norm_train_images = numpy.multiply(norm_train_images, 1.0 / 255.0)

    if not model:
        on_band_count = 0
        for b in band_list:
            if b == 1:
                on_band_count += 1

        model = model_for_type(neural_net_type, tile_size, on_band_count)

    model.fit(norm_train_images,
              npy_training_labels,
              n_epoch=number_of_epochs,
              shuffle=False,
              validation_set=.1,
              show_metric=True,
              run_id='mlp')

    return model


def model_for_type(neural_net_type, tile_size, on_band_count):
    """The neural_net_type can be: one_layer_relu,
                                   one_layer_relu_conv,
                                   two_layer_relu_conv."""
    network = tflearn.input_data(shape=[None, tile_size, tile_size, on_band_count])

    # NN architectures mirror ch. 3 of www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf
    if neural_net_type == 'one_layer_relu':
        network = tflearn.fully_connected(network, 64, activation='relu')
    elif neural_net_type == 'one_layer_relu_conv':
        network = conv_2d(network, 64, 12, strides=4, activation='relu')
        network = max_pool_2d(network, 3)
    elif neural_net_type == 'two_layer_relu_conv':
        network = conv_2d(network, 64, 12, strides=4, activation='relu')
        network = max_pool_2d(network, 3)
        network = conv_2d(network, 128, 4, activation='relu')
    else:
        print("ERROR: exiting, unknown layer type for neural net")

    # classify as road or not road
    softmax = tflearn.fully_connected(network, 2, activation='softmax')

    # hyperparameters based on www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf
    momentum = tflearn.optimizers.Momentum(
        learning_rate=.005, momentum=0.9,
        lr_decay=0.0002, name='Momentum')

    net = tflearn.regression(softmax, optimizer=momentum, loss='categorical_crossentropy')

    return tflearn.DNN(net, tensorboard_verbose=0)


def save_model(model, neural_net_type, bands, tile_size):
    """Save a DeepOSM tflearn model and its metadata. """
    model.save(CACHE_PATH + 'model.pickle')
    # dump the training metadata to disk, for later loading model from disk
    training_info = {'neural_net_type': neural_net_type,
                     'bands': bands,
                     'tile_size': tile_size}
    with open(CACHE_PATH + MODEL_METADATA_PATH, 'w') as outfile:
        pickle.dump(training_info, outfile)


def load_model(neural_net_type, tile_size, on_band_count):
    """Load the TensorFlow model serialized at path."""
    model = model_for_type(neural_net_type, tile_size, on_band_count)
    model.load(CACHE_PATH + 'model.pickle')
    return model


def list_findings(labels, test_images, model):
    """Return lists of predicted false negative/positive labels/data."""
    npy_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
    npy_test_images = npy_test_images.astype(numpy.float32)
    npy_test_images = numpy.multiply(npy_test_images, 1.0 / 255.0)

    false_pos = []
    fp_images = []
    index = 0
    for x in range(0, len(npy_test_images) - 100, 100):
        images = npy_test_images[x:x + 100]
        image_tuples = test_images[x:x + 100]
        index, false_pos, fp_images = sort_findings(model,
                                                    image_tuples,
                                                    images,
                                                    labels,
                                                    false_pos,
                                                    fp_images,
                                                    index)
    images = npy_test_images[index:]
    image_tuples = test_images[index:]
    index, false_pos, fp_images = sort_findings(model,
                                                image_tuples,
                                                images,
                                                labels,
                                                false_pos,
                                                fp_images,
                                                index)

    return false_pos, fp_images


def sort_findings(model, image_tuples, test_images, labels, false_positives, fp_images, index):
    """False positive if model says road doesn't exist, but OpenStreetMap says it does.

    False negative if model says road exists, but OpenStreetMap doesn't list it.
    """
    pred_index = 0
    for p in model.predict(test_images):
        label = labels[index][0]
        if has_ways_in_center(label, 1) and p[0] > .5:
            false_positives.append(p)
            fp_images.append(image_tuples[pred_index])
        # elif not has_ways_in_center(label, 16) and p[0] <= .5:
        #    false_negatives.append(p)
        #    fn_images.append(image_tuples[pred_index])
        pred_index += 1
        index += 1
    return index, false_positives, fp_images


def predictions_for_tiles(test_images, model):
    """Batch predictions on the test image set, to avoid a memory spike."""
    npy_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
    test_images = npy_test_images.astype(numpy.float32)
    test_images = numpy.multiply(test_images, 1.0 / 255.0)

    all_predictions = []
    for x in range(0, len(test_images) - 100, 100):
        for p in model.predict(test_images[x:x + 100]):
            all_predictions.append(p)

    for p in model.predict(test_images[len(all_predictions):]):
        all_predictions.append(p)
    assert len(all_predictions) == len(test_images)

    return all_predictions
