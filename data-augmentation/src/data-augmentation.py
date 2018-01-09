#!/usr/bin/env python

import numpy as np
from skimage import transform as trans
from skimage.util import crop
import matplotlib.image as mpimg
import argparse
import os
from fileutils import createName
from fileutils import readResolution
from fileutils import getSubfolders
from fileutils import getImageList
from imageUtils import loadJpegImage

DEFAULT_OUTPUT_PATH='./results/augmented'

def generateDescription():
    return """
        This tool performs data augmentation on all the images in the provided dataset folder. The data augmentation 
        techniques used are the following:
        
        - Image mirroring (1) 
        - Slight rotations (4)
        - Tone modifications (4)
        - Brightness modifications (4)
        - Cropping(5)
        
        The modifications are applied in a cascade-style (i.e.: each type of modification is applied to each of the
        images that resulted from the previous one) so the total number of new images per original image is 320. 
        
        The data-agumentation process will perform recursively in the provided directory, and the generated results
        folder will replicate the structure of the original.
        
        All the unmodified images will be also copied to the destination location.
    """

def defineParser():
    parser = argparse.ArgumentParser(description=generateDescription())
    parser.add_argument('imagePath', type=str, help='Path to the data directory')
    parser.add_argument('--output', dest='out', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Output directory for the cropped images')
    parser.add_argument('--outputRes', dest='outputRes', type=str,
                        help='Output resolution of the cropped images')

    return parser


def mirroring(image):
    return np.fliplr(image)


def toneModifications(img, rgb):
    modification = [np.full((img.shape[0], img.shape[1]), x) for x in rgb]

    _, _, channels = img.shape
    toneMask = np.zeros(img.shape)

    for i in range(channels):
        toneMask[:, :, i] = modification[i]

    return np.clip(img * toneMask, 0, 1)


def saveModification(originalImage, outputFolder, key, modifiedFile):
    filename = createName(originalImage, outputFolder, key)
    mpimg.imsave(filename, modifiedFile)


def slightRotation(img, degrees):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    rotation = trans.rotate(img, degrees)
    margin = np.ceil(np.abs((img.shape[0] * np.pi * degrees)/360))/2
    rotation = crop(rotation, ((margin, margin), (margin, margin), (0, 0)))
    rotation = trans.resize(rotation, (height, width, channels))
    return rotation


def applyTransformations(imageList, transformationTypes):
    results = imageList

    for transformationList in transformationTypes:
        workingSet = results

        for image in results:
            newImages = [f(image) for f in transformationList]
            workingSet = workingSet + newImages

        results = workingSet

    return results


def dataAugmentImage(img, imageName, outputFolder, augmentOptions):
    print('Augmenting data for ' + imageName + ' into ' + outputFolder)
    imageList = [
        img,
        mirroring(img)
    ]

    rotations = [
        lambda i: slightRotation(i, 8),
        lambda i: slightRotation(i, -8)
    ]

    tones = [
        lambda i: toneModifications(i, (1.1, 0.9, 1.1, 1.0)),
        lambda i: toneModifications(i, (0.9, 1.1, 1.1, 1.0)),
        lambda i: toneModifications(i, (1.1, 1.1, 0.9, 1.0)),
        lambda i: toneModifications(i, (0.9, 0.9, 0.9, 1.0)),
        lambda i: toneModifications(i, (1.1, 1.1, 1.1, 1.0))
    ]

    imageList = applyTransformations(imageList, [rotations, tones])

    for key, modification in enumerate(imageList):
        targetImage = modification

        if augmentOptions['outputRes']:
            resolution = readResolution(augmentOptions['outputRes'])
            newShape = (resolution[1], resolution[0], targetImage.shape[2])
            targetImage = trans.resize(targetImage, newShape)

        saveModification(imageName, outputFolder, key, targetImage)


def dataAugmentFolder(originalPath, folder, augmentOptions):
    originFolder = os.path.join(originalPath, folder)
    outputFolder = os.path.join(augmentOptions['out'], folder)

    pngList = getImageList(originFolder, "*.png")

    for image in pngList:
        loadedImage = mpimg.imread(image)
        dataAugmentImage(loadedImage, image, outputFolder, augmentOptions)

    jpgList = getImageList(originFolder, "*.jpg")

    for image in jpgList:
        cleanImage = loadJpegImage(image)
        cleanName = image.replace("jpg", "png")
        dataAugmentImage(cleanImage, cleanName, outputFolder, augmentOptions)


def createFolderStructure(outputPath, subfolders):
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    for folder in subfolders:
        outputFolder = os.path.join(outputPath, folder)

        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)


def dataAugment(imagePath, augmentOptions):
    subfolders = getSubfolders(imagePath)

    createFolderStructure(augmentOptions.out, subfolders)

    options = {
        'outputRes': augmentOptions.outputRes,
        'out': augmentOptions.out
    }

    for folder in subfolders:
        dataAugmentFolder(imagePath, folder, options)


def start():
    args = defineParser().parse_args()
    dataAugment(args.imagePath, args)

start()
