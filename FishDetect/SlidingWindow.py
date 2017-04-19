# import the necessary packages
import argparse
import time
import matplotlib.image as mping
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
from scipy import misc
# load the image and define the window width and height
image = img = skimage.io.imread("onefish.jpg")
(winW, winH) = (224, 224)

import os

os.environ["GLOG_minloglevel"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def pyramid(image, scale=1.5, minSize=(30, 30)):

    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = skimage.transform.resize(image, [w,w])

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image



# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW

        # since we do not have a classifier, we'll just draw the window
        clone = resized.copy()
        #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        plt.plot(1,2)
        time.sleep(0.025)