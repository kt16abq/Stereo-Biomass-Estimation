################################################################################
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
# from pylab import *
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from six.moves import cPickle as pickle
import tensorflow as tf

from caffe_classes import class_names

train_x = zeros((1, 227, 227, 2)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

################################################################################
# Read Image, and change to BGR


im1 = (imread("laska.png")[:, :, :3]).astype(float32)
im1 = im1 - mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
# im1 = imresize(im1,(227,227,3),'bilinear')
im1 = im1[:, :, 0:2]

im2 = (imread("H_801.png")[:, :, :3]).astype(float32)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]
im2 = imresize(im2, (227, 227, 3), 'bilinear')
im2 = im2[:, :, 0:2]
## load data
pickle_file = 'SHARKS.pickle'
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
image_size = 224
num_labels = 3
num_channels = 2  # g and b


def reformat(dataset, labels):
    dataset = dataset.reshape(

        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset[0:200], valid_labels[0:200])
test_dataset, test_labels = reformat(test_dataset[0:200], test_labels[0:200])
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
# input
batch_size = 15
patch_size = 3
depth = 8
num_hidden = 32


# judge
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
tf_valid_dataset = tf.constant(valid_dataset)
tf_test_dataset = tf.constant(test_dataset)


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


x = tf.placeholder(tf.float32, (None,) + xdim)

fc7W = tf.constant(net_data["fc7"][0])  # tf.truncated_normal(    [4096, 512], stddev=0.01)
fc7b = tf.constant(net_data["fc7"][1])
# tf.zeros([512])

fc8W = tf.Variable(tf.truncated_normal(
    [4096, 3], stddev=0.01))  # net_data["fc8"][0][:,0:3])
fc8b = tf.Variable(tf.zeros([3]))  # net_data["fc8"][1][0:3])


def model(data):
    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11 
    k_w = 11 
    c_o = 96 
    s_h = 4 
    s_w = 4
    conv1W = tf.constant(net_data["conv1"][0][:, :, 0:2, :])
    conv1b = tf.constant(net_data["conv1"][1])
    conv1_in = conv(data, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2 
    alpha = 2e-05 
    beta = 0.75 
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3 
    k_w = 3 
    s_h = 2 
    s_w = 2 
    padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5 
    k_w = 5 
    c_o = 256 
    s_h = 1 
    s_w = 1 
    group = 2
    conv2W = tf.constant(net_data["conv2"][0])
    conv2b = tf.constant(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2 
    alpha = 2e-05 
    beta = 0.75 
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3 
    k_w = 3 
    s_h = 2 
    s_w = 2 
    padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3 
    k_w = 3 
    c_o = 384 
    s_h = 1 
    s_w = 1 
    group = 1
    conv3W = tf.constant(net_data["conv3"][0])
    conv3b = tf.constant(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3 
    k_w = 3 
    c_o = 384 
    s_h = 1 
    s_w = 1 
    group = 2
    conv4W = tf.constant(net_data["conv4"][0])
    conv4b = tf.constant(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3 
    k_w = 3 
    c_o = 256 
    s_h = 1 
    s_w = 1 
    group = 2
    conv5W = tf.constant(net_data["conv5"][0])
    conv5b = tf.constant(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3 
    k_w = 3 
    s_h = 2 
    s_w = 2 
    padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # fc6
    # fc(4096, name='fc6')
    fc6W = tf.constant(net_data["fc6"][0])
    fc6b = tf.constant(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    # fc7
    # fc(4096, name='fc7')
    # fc7W = tf.constant(net_data["fc7"][0])
    # fc7b = tf.constant(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    # fc8
    # fc(1000, relu=False, name='fc8')
    # drop out!!!!!
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    return fc8


# prob
# softmax(name='prob'))

logits = model(tf_train_dataset)
prob = tf.nn.softmax(logits)
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

optimizer = tf.train.AdamOptimizer(0.00001).minimize(loss)

train_prediction = tf.nn.softmax(logits)
valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
test_prediction = tf.nn.softmax(model(tf_test_dataset))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print('Initialized')
num_steps = 10001
for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
    _, l, predictions = sess.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
        print('Minibatch loss at step %d: %f' % (step, l))
        print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        valid_prediction_res = sess.run(valid_prediction)
        print(valid_prediction_res)
        print(valid_labels)
        print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction_res, valid_labels))
print('Test accuracy: %.1f%%' % accuracy(sess.run(test_prediction), test_labels))

t = time.time()
output = sess.run(prob, feed_dict={x: [im1, im2]})
print(im1.shape)
################################################################################

# Output:


for input_im_ind in range(output.shape[0]):
    inds = argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print(class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]])

print(time.time() - t)
