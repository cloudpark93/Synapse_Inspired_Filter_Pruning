import numpy as np
from keras.datasets import mnist, cifar10
from keras.utils.np_utils import to_categorical
import os
from functools import partial

import tensorflow as tf

from config import config
from utils.image_processing import preprocess_image, resize_and_rescale_image

def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.compat.v1.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height
        # and width that is set dynamically by decode_jpeg. In other
        # words, the height and width of image is unknown at compile-i
        # time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).
        # The various adjust_* ops all require this range for dtype
        # float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def _parse_fn(example_serialized, is_training):
    """Helper function for parse_fn_train() and parse_fn_valid()

    Each Example proto (TFRecord) contains the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

    Args:
        example_serialized: scalar Tensor tf.string containing a
                            serialized Example protocol buffer.
        is_training: training (True) or validation (False).

    Returns:
        image_buffer: Tensor tf.string containing the contents of
        a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
    """
    feature_map = {
        'image/encoded': tf.compat.v1.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.compat.v1.FixedLenFeature([], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.compat.v1.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    parsed = tf.compat.v1.parse_single_example(example_serialized, feature_map)
    image = decode_jpeg(parsed['image/encoded'])

    if config.DATA_AUGMENTATION:
        image = preprocess_image(image, 224, 224, is_training=is_training)
    else:
        image = resize_and_rescale_image(image, 224, 224)
    # The label in the tfrecords is 1~1000 (0 not used).
    # So I think the minus 1 (of class label) is needed below.
    label = tf.one_hot(parsed['image/class/label'] - 1, 1000, dtype=tf.float32)
    return (image, label)

def normalize(x_train, x_test):
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    return x_train, x_test


def dataset(x):
    if x == 'mnist':
        # Load MNIST data set & reshape them from (n, 28, 28) to (n, 28, 28, 1)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print(x_train.shape)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

        # normalize input data from [0, 255] to [0, 1]
        # x_train = x_train.astype('float32') / 255.
        # x_test = x_test.astype('float32') / 255.
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = normalize(x_train, x_test)

        # converting y data into categorical (one-hot encoding)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Load test data and choose a random data out of the test data
        # choose a random data out of the test dataset, if you want to specifically use 'n'th data, x_input = x_test[[n]]
        x_random = np.random.choice(len(x_test), 1)
        x_random_input = x_test[x_random]  # x_test[[7983]]

        return x_train, x_test, y_train, y_test

    elif x == 'cifar10':
        # x_train.shape = (50000, 32, 32 ,3)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # convert the input data from integers to floats and normalize from [0, 255] to [0, 1]
        # x_train = x_train.astype('float32') / 255.
        # x_test = x_test.astype('float32') / 255.
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = normalize(x_train, x_test)

        # converting y data into categorical (one-hot encoding)
        num_classes = 10
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        # Load test data and choose a random data out of the test data
        # choose a random data out of the test dataset, if you want to specifically use 'n'th data, x_input = x_test[[n]]
        x_random = np.random.choice(len(x_test), 1)
        x_random_input = x_test[x_random]  # x_test[[7983]]

        return x_train, x_test, y_train, y_test

def get_dataset(tfrecords_dir, subset, batch_size):
    """Read TFRecords files and turn them into a TFRecordDataset."""
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    files = tf.compat.v1.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    shards = tf.compat.v1.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.compat.v1.cast(tf.shape(files)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=8192)
    
    parser = partial(
        _parse_fn, is_training=True if subset == 'train' else False)
    dataset = dataset.apply(
        tf.compat.v1.data.experimental.map_and_batch(
            map_func=parser,
            batch_size=batch_size,
            num_parallel_calls=AUTOTUNE))
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset