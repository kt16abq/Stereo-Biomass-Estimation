## method is from Udacity course

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import *
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import shutil

log_dir = '/tmp/tensorflow/Fish/logs/summaries'
try:
    shutil.rmtree(log_dir)
except:
    print(log_dir+' is empty!')
dropout = 0.5
batch_size = 60
depth = 8
num_hidden = 32
num_class = 3
## load data
pickle_file = 'FishSAlexNet.pickle'
net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

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
    print('Training set', train_dataset.shape, train_labels.shape, np.mean(train_dataset), np.std(train_dataset))
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

# reformat
def reformat(dataset, labels):
    labels = (np.arange(num_class) == labels[:, None]).astype(np.float32)
    return dataset, labels


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

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape, np.mean(train_dataset), np.std(train_dataset))
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# Input placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,  shape=(batch_size, 227,227,3), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(batch_size, num_class), name='y-input-Labels')
    tf.summary.image('input', x, 10)


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

#conv1
with tf.name_scope('conv1'):
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
with tf.name_scope('conv2'):
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
with tf.name_scope('convX3'):
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

#maxpool5
with tf.name_scope('maxpool5'):
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

with tf.name_scope('fcX3'):
    #fc6
    with tf.name_scope('fc6'):
        #fc(4096, name='fc6')
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    with tf.name_scope('dropout1'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability1', keep_prob)
        dropped1 = tf.nn.dropout(fc6, keep_prob)

    #fc7
    with tf.name_scope('fc7'):
        #fc(4096, name='fc7')
        fc7W = tf.Variable(tf.truncated_normal([4096, 1024], dtype=tf.float32, stddev=0.01),
                           trainable=True, name='weights')
        fc7b = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                           trainable=True, name='biases')
        fc7 = tf.nn.relu_layer(dropped1, fc7W, fc7b)
        variable_summaries(fc7)

    with tf.name_scope('dropout2'):
        tf.summary.scalar('dropout_keep_probability1', keep_prob)
        dropped2 = tf.nn.dropout(fc7, keep_prob)

    #fc8
    with tf.name_scope('fc8'):
        #fc(1000, relu=False, name='fc8')
        fc8W = tf.Variable(tf.truncated_normal([1024, num_class], dtype=tf.float32, stddev=0.01),
                           trainable=True, name='weights')
        fc8b = tf.Variable(tf.constant(0.0, shape=[num_class], dtype=tf.float32),
                           trainable=True, name='biases')
        y = tf.nn.xw_plus_b(dropped2, fc8W, fc8b, name='y')
        variable_summaries(y)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(y,name='y')



# retain: from pooling to fc




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
    variable_summaries(y)
    #variable_summaries(diff)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.000002).minimize(cross_entropy, var_list=[fc7W, fc7b, fc8W, fc8b])

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to
# /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
merged = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

print('init..')
init = tf.initialize_all_variables()

sess.run(init)
#init_vars_op = tf.variables_initializer([fcw,fcb])
#sess.run(init_vars_op)

# Train the model, and also write summaries.
# Every 10th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries
#print(tf.global_variables())

for i in range(10001):
    offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # print(batch_data[0])
    # print("max,min", np.max(batch_data[0]), np.min(batch_data[0]))
    if i % 10 == 0:  # Record summaries and valid-set accuracy
        feed_dict = {x: valid_dataset, y_: valid_labels, keep_prob: 1}
        print("max,min", np.max(valid_dataset[0]), np.min(valid_dataset[0]))
        summary, acc,ap = sess.run([merged, accuracy,fc6], feed_dict=feed_dict)
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
        print('ap shape: ', ap.shape)
        print("max,min", np.max(ap[0]), np.min(ap[0]))
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