#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from fileutils import readAspect
import matplotlib.patches as patches

INPUTMODEL='./results/object_convnet1/1515141257'

AVG_WIDTH=0.2
VAR_WIDTH=0.1
VAR_POSITION=0.3
ASPECT_RATIO='5:6'
TOLERANCE=0.65
MIN_WIDTH=0.01

def getInputData(imagePath):
    image = mpimg.imread(imagePath)
    normalized = np.asarray(image[:, :, :3], dtype=np.float32)
    return normalized

def predictImg(inputData):
    with tf.Session(graph=tf.Graph()) as sess:
        output = tf.get_default_graph().get_tensor_by_name('softmax_tensor:0')
        preds = sess.run(output, feed_dict={"input_tensors:0": inputData})

        return preds[1] > TOLERANCE

def viewImageWithBoxes(image, boxes):
    _, ax = plt.subplots(1)

    ax.imshow(image)

    for box in boxes:
        rect = patches.Rectangle((box['x'] -box['width']/2, box['y'] - box['height']/2),
                                 box['width'], box['height'],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def createBoxes(image, n):
    height, width, _ = image.shape

    boxes = []
    for i in range(0, n):
        box = {
            'width': int(max(np.random.normal(AVG_WIDTH*width, VAR_WIDTH*width), width*MIN_WIDTH))
        }

        box['height'] = int(box['width']*readAspect(ASPECT_RATIO))

        box['x'] = int(min(
            max(np.random.normal(width/2, VAR_POSITION*width), box['width']/2),
            width - box['width']/2 - 1))

        box['y'] = int(min(
            max(np.random.normal(height/2, VAR_POSITION*height), box['height']/2),
            height - box['height']/2 - 1))

        boxes.append(box)

    return boxes

def generateBoxingForImage(imagePath):
    image = getInputData(imagePath)

    print('Shape of the image: ' + str(image.shape))
    boxes = createBoxes(image, 15)

    print('Boxes: ' + str(boxes))
    viewImageWithBoxes(image, boxes)


generateBoxingForImage('../../examples/2016-08-26 00.26.17.png')

