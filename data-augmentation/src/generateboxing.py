#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import transform as trans
from fileutils import readAspect
import matplotlib.patches as patches
import argparse
from fileutils import getImageList
from fileutils import generateName
import os
from imageUtils import loadJpegImage

DEFAULT_OUTPUT_PATH='./results/selection'

AVG_WIDTH=0.3
VAR_WIDTH=0.2
VAR_POSITION=0.4
ASPECT_RATIO='5:6'
TOLERANCE=0.50
MIN_WIDTH=0.02
INPUT_SHAPE=(128, 96, 3)

FOLDER_POSITIVE='positive'
FOLDER_NEGATIVE='negative'


def predictImg(inputData, model):
    with tf.Session(graph=tf.Graph()) as sess:
        _ = tf.saved_model.loader.load(sess, ['serve'], model)
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


def getSubImage(image, box):
    xi = int(box['x'] - box['width']/2)
    xf = int(box['x'] + box['width']/2)
    yi = int(box['y'] - box['height']/2)
    yf = int(box['y'] + box['height']/2)

    subimage = image[yi:yf, xi:xf, :]
    subimage = trans.resize(subimage, INPUT_SHAPE)

    return subimage


def recognizeInBoxes(image, boxes, model):
    predictions = []

    for i in range(0, len(boxes)):
        subimage = getSubImage(image, boxes[i])
        predictions.append(predictImg(subimage, model))

    return predictions


def createDataset(image, predictions):
    initialTargets = [box for i, box in enumerate(boxes) if predictions[i] > TOLERANCE]
    viewImageWithBoxes(image, initialTargets)
    print(predictions)


def cropImages(image, boxes, predictions, outputFolder):
    print('Cropping')

    imageName = generateName()

    for i in range(0, len(boxes)):
        subimage = getSubImage(image, boxes[i])

        if predictions[i] > TOLERANCE:
            imagePath = os.path.join(outputFolder, FOLDER_POSITIVE, imageName + str(i) + ".png")
        else:
            imagePath = os.path.join(outputFolder, FOLDER_NEGATIVE, imageName + str(i) + ".png")

        mpimg.imsave(imagePath, subimage)


def highlight(image, boxes, predictions, outputFolder):
    imageShape = image.shape
    mask = np.zeros(imageShape)

    for i in range(0, len(boxes)):
        xi = int(boxes[i]['x'] - boxes[i]['width']/2)
        xf = int(boxes[i]['x'] + boxes[i]['width']/2)
        yi = int(boxes[i]['y'] - boxes[i]['height']/2)
        yf = int(boxes[i]['y'] + boxes[i]['height']/2)

        if predictions[i] > TOLERANCE:
            mask[yi:yf, xi:xf, :] = mask[yi:yf, xi:xf, :] + 1
        else:
            mask[yi:yf, xi:xf, :] = mask[yi:yf, xi:xf, :] - 2


    minimumValue = np.amin(mask)
    maximumValue = np.amax(mask)
    mask = (mask - minimumValue)/(maximumValue - minimumValue)

    imageName = generateName()
    imagePath = os.path.join(outputFolder, imageName + ".png")
    mpimg.imsave(imagePath, mask)


def generateBoxingForImage(imagePath, options):
    image = loadJpegImage(imagePath)

    print('Shape of the image: ' + str(image.shape))
    boxes = createBoxes(image, options.boxNumber)
    predictions = recognizeInBoxes(image, boxes, options.model)

    if (options.task == 'dataset'):
        createDataset(image, predictions)
    elif (options.task == 'cropping'):
        cropImages(image, boxes, predictions, DEFAULT_OUTPUT_PATH)
    elif (options.task == 'highlighting'):
        highlight(image, boxes, predictions, DEFAULT_OUTPUT_PATH)


def generateDescription():
    return """
        This tool generates a random boxing pattern for a set of images, running an object recognizing algorithm
        for each patch corresponding to a box. This boxing division of the image can be used in two tasks:
        
        - Creation of datasets of images labeled with object location and boxing.
        - Creation of image patches divided according to the algorithm classification.
        - Highlight masks for the selected objects, based on box superposition
        
        This second task is mainly thought as a mean to extend the original object recognition dataset, by choosing, 
        from the division performed by the algorithm, the false positives and false negatives, that make interesting
        cases that may improve the algorithm's performance in real data.
    """


def defineParser():
    parser = argparse.ArgumentParser(description=generateDescription())

    parser.add_argument('task', type=str, help='Task to perform. Available values: dataset or cropping')
    parser.add_argument('imagePath', type=str, help='Path to the data directory')
    parser.add_argument('boxNumber', type=int, help='Number of boxes to be generated')
    parser.add_argument('model', type=str, help='Path to the directory containing the model to load')

    return parser


def createOutputStructure(outputPath):
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    positiveFolder = os.path.join(outputPath, FOLDER_POSITIVE)

    if not os.path.exists(positiveFolder):
        os.mkdir(positiveFolder)

    negativeFolder = os.path.join(outputPath, FOLDER_NEGATIVE)

    if not os.path.exists(negativeFolder):
        os.mkdir(negativeFolder)


def start():
    args = defineParser().parse_args()

    createOutputStructure(DEFAULT_OUTPUT_PATH)

    images = getImageList(args.imagePath, "*.jpg")

    for image in images:
        generateBoxingForImage(image, args)

start()

