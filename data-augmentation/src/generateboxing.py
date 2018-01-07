#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import transform as trans
from fileutils import readAspect
import matplotlib.patches as patches

MODEL_ID='1515339063'
INPUTMODEL='./results/object_convnet1/' + MODEL_ID

AVG_WIDTH=0.2
VAR_WIDTH=0.1
VAR_POSITION=0.3
ASPECT_RATIO='5:6'
TOLERANCE=0.73
MIN_WIDTH=0.05
INPUT_SHAPE=(128, 96, 3)

def getInputData(imagePath):
    image = mpimg.imread(imagePath)
    normalized = np.asarray(image[:, :, :3], dtype=np.float32)
    return normalized

def predictImg(inputData):
    with tf.Session(graph=tf.Graph()) as sess:
        _ = tf.saved_model.loader.load(sess, ['serve'], INPUTMODEL)
        output = tf.get_default_graph().get_tensor_by_name('softmax_tensor:0')
        preds = sess.run(output, feed_dict={"input_tensors:0": inputData})

        return preds[0, 1]

def viewImageWithBoxes(image, boxes):
    _, ax = plt.subplots(1)

    ax.imshow(image)

    for index, box in enumerate(boxes):
        color = 'g'

        rect = patches.Rectangle((box['x'] -box['width']/2, box['y'] - box['height']/2),
                                 box['width'], box['height'],
                                 linewidth=1, edgecolor=color, facecolor='none')
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

def recognizeInBoxes(image, boxes):
    predictions = []

    for i in range(0, len(boxes)):
        xi = int(boxes[i]['x'] - boxes[i]['width']/2)
        xf = int(boxes[i]['x'] + boxes[i]['width']/2)
        yi = int(boxes[i]['y'] - boxes[i]['height']/2)
        yf = int(boxes[i]['y'] + boxes[i]['height']/2)

        subimage = image[yi:yf, xi:xf, :]
        subimage = trans.resize(subimage, INPUT_SHAPE)
        predictions.append(predictImg(subimage))

    return predictions

def generateBoxingForImage(imagePath):
    image = getInputData(imagePath)

    print('Shape of the image: ' + str(image.shape))
    boxes = createBoxes(image, 100)
    predictions = recognizeInBoxes(image, boxes)

    initialTargets = [box for i, box in enumerate(boxes) if predictions[i] > TOLERANCE]

    viewImageWithBoxes(image, initialTargets)

    print(predictions)



generateBoxingForImage('../../examples/2016-08-26 00.26.17.png')

