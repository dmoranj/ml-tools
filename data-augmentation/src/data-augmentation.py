#!/usr/bin/env python

import argparse
import os
import re
import glob

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

def dataAugmentImage(image, outputFolder):
    print('Augmenting data for ' + image + ' into ' + outputFolder)

def dataAugmentFolder(originalPath, folder, augmentOptions):
    originFolder = os.path.join(originalPath, folder)
    outputFolder = os.path.join(augmentOptions.out, folder)

    imageList = getImageList(originFolder)

    for image in imageList:
        dataAugmentImage(image, outputFolder)

def dataAugment(imagePath, augmentOptions):
    subfolderList = getSubfolders(imagePath)

    for folder in subfolderList:
        dataAugmentFolder(imagePath, folder, augmentOptions)


def start():
    args = defineParser().parse_args()
    dataAugment(args.imagePath, args)

start()
