## method is from Udacity course

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from scipy.misc import imresize
import shutil

log_dir = '/tmp/tensorflow/FishVGG/logs/summaries'
weight_file = 'vgg16_weights.npz'
try:
    shutil.rmtree(log_dir)
except:
    print(log_dir+' is empty!')
dropout = 0.5
batch_size = 60
patch_size = 9
depth = 8
num_hidden = 32
num_class = 3
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
    print('Training set', train_dataset.shape, train_labels.shape, np.mean(train_dataset), np.std(train_dataset))
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
# reformat
image_size = 227
image_size_model = 224
num_labels = 3
num_channels = 3  # g and b


def reformat(dataset, labels):
    dataset_reshape = np.ndarray(shape=(dataset.shape[0], image_size_model, image_size_model, num_channels),
                         dtype=np.float32)
    for k in range(dataset.shape[0]):
        img = dataset[k,:, :, :]
        # img = imresize(img,[image_size_model,image_size_model])
        dataset_reshape[k,:, :, :]=img[0:224,0:224,:]
        k = k+1
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset_reshape, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape, np.mean(train_dataset), np.std(train_dataset))
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
sess = tf.InteractiveSession()
# Create a multilayer model.

# Input placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=(batch_size, image_size_model, image_size_model, num_channels), name='x-input-Pictures')
    y_ = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='y-input-Labels')
    tf.summary.image('input', x, 10)

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


'''VGG16'''
parameters = []
# conv1_1
with tf.name_scope('conv1_1') as scope:
    tf.summary.histogram('input', x)
    kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]
# conv1_2
with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]
    tf.summary.histogram('conv1-result', conv1_2)
# pool1
pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

# conv2_1
with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]

# conv2_2
with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]
    tf.summary.histogram('conv2-result', conv2_2)

# pool2
pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

# conv3_1
with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]

# conv3_2
with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]

# conv3_3
with tf.name_scope('conv3_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]
    tf.summary.histogram('conv3-result', conv3_3)

# pool3
pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

# conv4_1
with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]

# conv4_2
with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]

# conv4_3
with tf.name_scope('conv4_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]
    tf.summary.histogram('conv4-result', conv4_3)

# pool4
pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

# conv5_1
with tf.name_scope('conv5_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]

# conv5_2
with tf.name_scope('conv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]

# conv5_3
with tf.name_scope('conv5_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=False, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(out, name=scope)
    parameters += [kernel, biases]
    tf.summary.histogram('conv5-result', conv5_3)

# pool5
pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

# fc1
with tf.name_scope('fc1') as scope:
    shape = int(np.prod(pool5.get_shape()[1:]))
    fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1), trainable=False, name='weights')
    fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=False, name='biases')
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
    fc1 = tf.nn.relu(fc1l)
    parameters += [fc1w, fc1b]
    tf.summary.histogram('fc1-result', fc1)

# fc2 -retrain
with tf.name_scope('dropout1'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability1', keep_prob)
    dropped1 = tf.nn.dropout(fc1, keep_prob)
with tf.name_scope('fc2') as scope:
    fc2w = tf.Variable(tf.truncated_normal([4096, 128], dtype=tf.float32, stddev=0.01), trainable=True, name='weights')
    fc2b = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
    fc2l = tf.nn.bias_add(tf.matmul(dropped1, fc2w), fc2b)
    fc2 = tf.nn.relu(fc2l)
    parameters += [fc2w, fc2b]
    variable_summaries(fc2b)
    tf.summary.histogram('fc2-result', fc2)

# fc3 -retrain
with tf.name_scope('dropout'):
    dropped = tf.nn.dropout(fc2, keep_prob)

with tf.name_scope('fc3') as scope:
    fc3w = tf.Variable(tf.truncated_normal([128, num_class], dtype=tf.float32, stddev=0.01), trainable=True, name='weights')
    fc3b = tf.Variable(tf.constant(0.0, shape=[num_class], dtype=tf.float32), trainable=True, name='biases')
    y = tf.nn.bias_add(tf.matmul(dropped, fc3w), fc3b)
    parameters += [fc3w, fc3b]


# load weights
weights = np.load(weight_file)
keys = sorted(weights.keys())
for i, k in enumerate(keys):
    print(i, k, np.shape(weights[k]))
    try:
        sess.run(parameters[i].assign(weights[k]))
    except:
        print(i, k, np.shape(weights[k]) , 'is the random start')


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
    train_step = tf.train.AdamOptimizer(0.00002).minimize(cross_entropy, var_list=[fc2w, fc2b, fc3w, fc3b])

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
