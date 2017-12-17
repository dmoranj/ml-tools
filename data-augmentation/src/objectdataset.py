import os
import logging
import glob
import pandas as pd
import numpy as np
import matplotlib.image as mpimg

DEFAULT_POSITIVE_FOLDER = 'Positive'
DEFAULT_NEGATIVE_FOLDER = 'Negative'

def readImages(imageList):

    images = []

    for imagePath in imageList:
        image = mpimg.imread(imagePath)
        normalized = np.asarray(image[:, :, :3], dtype=np.float16)
        images.append(normalized)

    return np.array(images)

def generateObservations(outputFolder, positiveFolder, negativeFolder, maxResults):
    positiveFolder = os.path.join(outputFolder, positiveFolder)
    negativeFolder = os.path.join(outputFolder, negativeFolder)

    imageList = []

    logging.info('Reading positive test case folder')
    for key, imagePath in enumerate(glob.glob(os.path.join(positiveFolder, "*.png"))):
        if key > maxResults:
            break

        imageList.append((imagePath, 1.0))

    logging.info('Reading negative test case folder')
    for key, imagePath in enumerate(glob.glob(os.path.join(negativeFolder, "*.png"))):
        if key > maxResults:
            break

        imageList.append((imagePath, 0.0))

    imageDf = pd.DataFrame(data=imageList, columns=['images', 'labels'])

    return imageDf


def loadImageSet(outputFolder, trainTestSplit, maxResults=7000):

    observations = generateObservations(outputFolder, DEFAULT_POSITIVE_FOLDER, DEFAULT_NEGATIVE_FOLDER, maxResults)

    trainObservations = observations.sample(frac=trainTestSplit)
    testObservations = observations.sample(frac=1 - trainTestSplit)

    imageSet = {
        'train': {
            'images': readImages(trainObservations['images'].tolist()),
            'labels': trainObservations.as_matrix(['labels'])
        },
        'test': {
            'images': readImages(testObservations['images'].tolist()),
            'labels': testObservations.as_matrix(['labels'])
        }
    }

    return imageSet

