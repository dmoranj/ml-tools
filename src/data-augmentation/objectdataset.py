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

    return imageDf.sample(frac=1)


def loadImageSet(inputFolder, maxResults = 7000):

    subfolders = [dir for dir in os.listdir(inputFolder) if os.path.isdir(os.path.join(inputFolder, dir))]
    imageSet = {}

    for subfolder in subfolders:
        folderPath = os.path.join(inputFolder, subfolder)
        observations = generateObservations(folderPath, DEFAULT_POSITIVE_FOLDER, DEFAULT_NEGATIVE_FOLDER, maxResults)

        imageSet[subfolder] = {
            'images': readImages(observations['images'].tolist()),
            'labels': observations.as_matrix(['labels'])
        }

    return imageSet



