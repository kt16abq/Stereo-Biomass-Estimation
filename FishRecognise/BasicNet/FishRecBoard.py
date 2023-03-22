## method is from Stereo camera

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import shutil

log_dir = '/tmp/tensorflow/Fish/logs/summaries'
shutil.rmtree(log_dir)
dropout = 0.5
batch_size = 60
patch_size = 9
depth = 8
num_hidden = 32
## load data
pickle_file = 'FishS.pickle'
print("Loading data..")
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
# reformat
image_size = 227
num_labels = 3
num_channels = 3  # g and b


def reformat(dataset, labels):
    dataset = dataset.reshape(
        #  (-1, image_size*image_size*num_channels)).astype(np.float32)
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
sess = tf.InteractiveSession()
# Create a multilayer model.

# Input placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels), name='x-input-Pictures')
    y_ = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='y-input-Labels')
    tf.summary.image('input', x, 10)


# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def cnn_layer(input_tensor, core_dim, input_depth, output_depth, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([core_dim, core_dim, input_depth, output_depth])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_depth])
            variable_summaries(biases)
        with tf.name_scope('Convolution'):
            preactivate = tf.nn.conv2d(input_tensor, weights, [1, 2, 2, 1], padding='SAME') + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def fc_layer(input_tensor, input_depth, output_depth, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_depth, output_depth])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_depth])
            variable_summaries(biases)
        shape = input_tensor.get_shape().as_list()
        reshape = tf.reshape(input_tensor, [shape[0], -1])
        with tf.name_scope('layer'):
            preactivate = tf.matmul(reshape, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate)
        return activations


hidden1 = cnn_layer(x, patch_size, num_channels, depth, 'cnn-layer1')
hidden2 = cnn_layer(hidden1, patch_size, depth, depth, 'cnn-layer2')
hidden3 = fc_layer(hidden2, (image_size // 4 + 1) * (image_size // 4 + 1) * depth, num_hidden, 'fc-layer3')
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden3, keep_prob)

# Do not apply softmax activation yet, see below.
y = fc_layer(dropped, num_hidden, num_labels, 'fc-layer4',act=tf.identity)

with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.00002).minimize(
        cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to
# /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
tf.global_variables_initializer().run()

# Train the model, and also write summaries.
# Every 10th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries


for i in range(10001):
    offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    if i % 10 == 0:  # Record summaries and valid-set accuracy
        feed_dict = {x: valid_dataset, y_: valid_labels, keep_prob: 1}
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
        feed_dict = {x: batch_data, y_: batch_labels, keep_prob: 0.5}
        if i % 100 == 99:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict,
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
        else:  # Record a summary
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()
