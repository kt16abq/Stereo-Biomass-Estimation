from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
# from MyAlexnet import *
from tensorflow.contrib.slim.nets import alexnet

slim = tf.contrib.slim
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
num_channels = 2 # g and b
def reformat(dataset, labels):
  dataset = dataset.reshape(
    #  (-1, image_size*image_size*num_channels)).astype(np.float32)
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
# judge
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


batch_size = 8
patch_size = 3
depth = 8
num_hidden = 32

graph = tf.Graph()

# test Alexnet
with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    logits, end_points = alexnet.alexnet_v2(test_dataset, num_classes=3, is_training=False, dropout_keep_prob=0.5,
                              spatial_squeeze=True, scope='alexnet_v2')
    print(logits.get_shape())
    predictions = tf.argmax(logits, 1)
    print(predictions.get_shape())
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(logits))
num_steps = 10001
#
# with tf.Session(graph=graph) as session:
#     tf.global_variables_initializer().run()
#     print('Initialized')
#     for step in range(num_steps):
#         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#         batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#         batch_labels = train_labels[offset:(offset + batch_size), :]
#         feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
#         _, l, predictions = session.run(
#             [optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if (step % 50 == 0):
#             print('Minibatch loss at step %d: %f' % (step, l))
#             print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#             print('Validation accuracy: %.1f%%' % accuracy(
#                 valid_prediction.eval(), valid_labels))
#     print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))