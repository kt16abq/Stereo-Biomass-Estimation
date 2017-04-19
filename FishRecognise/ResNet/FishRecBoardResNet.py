## method is from Udacity course

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import skimage.transform
from six.moves import cPickle as pickle
from six.moves import range
from scipy.misc import imresize
import shutil

log_dir = '/tmp/tensorflow/Fish/logs/summaries'
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
pickle_file = 'FishSResNet.pickle'
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
# Create a multilayer model.
sess = tf.InteractiveSession()

# load current structure
new_saver = tf.train.import_meta_graph('ResNet-L50.meta')
new_saver.restore(sess, './ResNet-L50.ckpt')

graph = tf.get_default_graph()

# add transfer learning layer
pooling_tensor = graph.get_tensor_by_name("avg_pool:0")

# Input placeholders
with tf.name_scope('input'):
    x = graph.get_tensor_by_name("images:0")
    y_ = tf.placeholder(tf.float32, shape=(batch_size, num_class), name='y-input-Labels')
    tf.summary.image('input', x, 10)


with tf.name_scope('avg_pooling'):
    variable_summaries(pooling_tensor)
# retain: from pooling to fc
with tf.name_scope('dropout1'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability1', keep_prob)
    dropped = tf.nn.dropout(pooling_tensor, keep_prob)
with tf.name_scope('trans_fc') as scope:
    fcw = tf.Variable(tf.truncated_normal([2048, num_class], dtype=tf.float32, stddev=0.01), trainable=True, name='weights')
    fcb = tf.Variable(tf.constant(0.0, shape=[num_class], dtype=tf.float32), trainable=True, name='biases')
    y = tf.nn.bias_add(tf.matmul(dropped, fcw), fcb)
    variable_summaries(y)



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
    train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy, var_list=[fcw, fcb])

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

print('init..')
tf.global_variables_initializer().run()
new_saver.restore(sess, './ResNet-L50.ckpt')
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
        summary, acc,ap = sess.run([merged, accuracy,pooling_tensor], feed_dict=feed_dict)
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