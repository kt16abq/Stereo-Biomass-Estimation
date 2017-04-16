## method is from Udacity course
# package file and save

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
#from IPython.display import display, Image
import matplotlib.image as mpimg
#from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

train_folders = ['Fish_data_train/Lethrinus', 'Fish_data_train/Lutjanus', 'Fish_data_train/Plectropomus']
test_folders = ['Fish_data_test/Lethrinus', 'Fish_data_test/Lutjanus', 'Fish_data_test/Plectropomus']
valid_folders = ['Fish_data_valid/Lethrinus', 'Fish_data_valid/Lutjanus', 'Fish_data_valid/Plectropomus']

image_size = 227  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
color_chanel = 3
def load_class(data_folders, min_num_images):
  """Load the data for a single class label."""
  dataset = np.ndarray(shape=(min_num_images, image_size, image_size, color_chanel),
                         dtype=np.float32)
  label = np.ndarray(shape=(min_num_images), dtype=np.float32)
  num_images = 0
  num_label = 0
  for folder in data_folders:
    image_files = os.listdir(folder)
    print(folder)
    for image in image_files:
      image_file = os.path.join(folder, image)
      print(image_file)
      try:
        image_data = (mpimg.imread(image_file).astype(float)-0.345)/0.276/10
        # image_data = image_data[0:224,0:224,:]
        # image_data_g = image_data[:, :, 1]
        # image_data_b = image_data[:, :, 2]
        # check
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #x = y = np.arange(0, 224, 1)
        #X, Y = np.meshgrid(x, y)
        #Z = image_data[:, :, 0]
        #print(Z.shape)
        #ax.plot_surface(X, Y, Z)
        #ax.set_xlabel('g')
        #ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')
        #plt.show()
        if image_data.shape != (image_size, image_size,3):
          raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset[num_images, :, :,:] = image_data
        # dataset[num_images, :, :,0] = image_data_g
        # dataset[num_images, :, :, 1] = image_data_b
        label[num_images] = num_label
        num_images = num_images + 1
      except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    num_label = num_label+1

  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset,label

train_dataset, train_labels = load_class(train_folders, 300)
test_dataset, test_labels = load_class(test_folders, 60)
valid_dataset, valid_labels = load_class(valid_folders, 60)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

## data ready
pickle_file = os.path.join('.', 'FishS.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)