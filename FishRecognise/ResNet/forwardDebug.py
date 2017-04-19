import os
os.environ["GLOG_minloglevel"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf
import shutil
# returns the top1 string


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img


log_dir = '/tmp/tensorflow/Fish/logs/summaries'
try:
    shutil.rmtree(log_dir)
except:
    print(log_dir+' is empty!')
num_class = 3

img = load_image("data/fish.png")

sess = tf.Session()

new_saver = tf.train.import_meta_graph('ResNet-L50.meta')
new_saver.restore(sess, './ResNet-L50.ckpt')


graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
pooling_tensor = graph.get_tensor_by_name("avg_pool:0")
# retain: from pooling to fc
with tf.name_scope('trans_fc') as scope:
    fcw = tf.Variable(tf.truncated_normal([2048, num_class], dtype=tf.float32, stddev=0.01), trainable=True, name='weights')
    fcb = tf.Variable(tf.constant(0.0, shape=[num_class], dtype=tf.float32), trainable=True, name='biases')
    y = tf.nn.bias_add(tf.matmul(pooling_tensor, fcw), fcb)

train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

images = graph.get_tensor_by_name("images:0")


#init = tf.initialize_all_variables()
#sess.run(init)
print( "graph restored")

batch = img.reshape((1, 224, 224, 3))
feed_dict = {images: batch}

ap = sess.run(pooling_tensor, feed_dict=feed_dict)

#print(prob)
#print(prob.shape)

print(ap)
print(ap.shape)
print("max,min",np.max(ap[0]),np.min(ap[0]))

train_writer.close()