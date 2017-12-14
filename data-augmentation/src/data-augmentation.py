#!/usr/bin/env python

import numpy as np
import matplotlib.image as mpimg
import argparse
import os
import re
import glob
from fileutils import createName

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

def getSubfolders(path):
    return [re.sub(r"^/", '', x[0].replace(path, '')) for x in os.walk(path)]

def getImageList(folder):
    return glob.glob(os.path.join(folder, "*.png"))

def mirroring(image):
    return np.fliplr(image)

def toneModifications(img, rgb):
    modification = [np.full((img.shape[0], img.shape[1]), x) for x in rgb]

    toneMask = np.zeros(img.shape)

    for i in range(4):
        toneMask[:, :, i] = modification[i]

    return np.clip(img * toneMask, 0, 1)

def saveModification(originalImage, outputFolder, key, modifiedFile):
    filename = createName(originalImage, outputFolder, key)
    mpimg.imsave(filename, modifiedFile)

def dataAugmentImage(image, outputFolder):
    print('Augmenting data for ' + image + ' into ' + outputFolder)
    img = mpimg.imread(image)
    imageList = [
        img,
        mirroring(img),
        toneModifications(img, (1.1, 0.9, 1.1, 1.0))
    ]

    for key, modification in enumerate(imageList):
        saveModification(image, outputFolder, key, modification)


def dataAugmentFolder(originalPath, folder, augmentOptions):
    originFolder = os.path.join(originalPath, folder)
    outputFolder = os.path.join(augmentOptions.out, folder)

    imageList = getImageList(originFolder)

    for image in imageList:
        dataAugmentImage(image, outputFolder)

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

    for folder in subfolders:
        dataAugmentFolder(imagePath, folder, augmentOptions)


def start():
    args = defineParser().parse_args()
    dataAugment(args.imagePath, args)

start()
