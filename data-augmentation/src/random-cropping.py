#!/usr/bin/env python
import matplotlib.image as mpimg
import numpy as np
import argparse
import os
import glob
from skimage import transform as trans
from fileutils import readAspect
from fileutils import readResolution
from fileutils import createName
from imageUtils import createPngImage

DEFAULT_NUM_CROPS=10

DEFAULT_HEIGHT = 40
DEFAULT_WIDTH = 40
DEFAULT_VARIANCE = 20
DEFAULT_OUTPUT_PATH = './results'

CANDIDATE_FOLDER = 'candidates'
DATA_FOLDER = 'data'

def generateDescription():
    return """
        The purpose of these tool is to process all the images in a dataset, generating, for each one of them
        a set of cropped subimages, expecting to capture interesting objects to be labeled in the process.
        
        The tool lets the user tune the general parameters of the random cropping, so to maximize the possibility
        of generating cropped images that are interesting for the task at hand. It will also help in making the
        dataset homogeneous, by down or upsizing each of the cropped segments to a predetermined size if required.
        
        The cropping tool will generate an output directory (defaulting to "/data") whith a subfolder "/candidates"
        containing all the cropped images.
        
        If the --aspect option is used, the height parameter will be ignored, and the frame height will be adapted
        to the calculated width of the image.
        """

def defineParser():
    parser = argparse.ArgumentParser(description=generateDescription())
    parser.add_argument('imagePath', type=str, help='Path to the data directory')
    parser.add_argument('--height', dest='height', type=int, default=DEFAULT_HEIGHT,
                        help='Mean height of the cropped segment (percentage)')
    parser.add_argument('--width', dest='width', type=int, default=DEFAULT_WIDTH,
                        help='Mean width of the cropped segment (percentage)')
    parser.add_argument('--variance', dest='var', type=float, default=DEFAULT_VARIANCE,
                        help='Variance for the random cropping size distribution')
    parser.add_argument('--output', dest='out', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Output directory for the cropped images')
    parser.add_argument('--aspect', dest='aspect', type=str,
                        help='Aspect ratio of the cropping frame')
    parser.add_argument('--crops', dest='crops', type=int, default=DEFAULT_NUM_CROPS,
                        help='Number of crops to create for each image')
    parser.add_argument('--outputRes', dest='outputRes', type=str,
                        help='Output resolution of the cropped images')

    return parser

def getImageList(inputPath):
  return glob.glob(os.path.join(inputPath, "*.jpg"))

def createOutputStructure(outputPath):
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    candidateFolder = os.path.join(outputPath, CANDIDATE_FOLDER)

    if not os.path.exists(candidateFolder):
        os.mkdir(candidateFolder)

    dataFolder = os.path.join(outputPath, DATA_FOLDER)

    if not os.path.exists(dataFolder):
        os.mkdir(dataFolder)

def createCropFrame(shape, cropParams):
    randomVariation = np.clip(np.random.normal(1, cropParams['var'], (2, 1)), 0.5, 1.5)

    height = shape[0]*cropParams['height']*randomVariation[0]
    width = shape[1]*cropParams['width']*randomVariation[1]

    if (cropParams['aspect']):
        height = width*cropParams['aspect']

    return (int(height[0]), int(width[0]))

def cropImageWithFrame(img, frame):
    availablePositions = (img.shape[0] - frame[0], img.shape[1] - frame[1])
    randomVariation = np.random.uniform(size=(1, 2))

    origin = np.floor(np.array(availablePositions)*randomVariation).astype(int)
    end = origin + frame

    origin = origin[0]
    end = end[0]

    newImage = img[origin[0]:end[0], origin[1]:end[1], :]

    return newImage


def cropImage(image, inputPath, outputPath, cropParams):
    print('Cropping image ' + image)
    img = mpimg.imread(image)

    for i in range(cropParams['crops']):
        cropFrame = createCropFrame(img.shape, cropParams)
        newImage = cropImageWithFrame(img, cropFrame)
        cropName = createName(image, outputPath, i)

        if cropParams['outputRes']:
            newShape = (cropParams['outputRes'][1], cropParams['outputRes'][0], newImage.shape[2])
            newImage = trans.resize(newImage, newShape)

        mpimg.imsave(cropName, newImage)


def randomCrop(inputPath, outputPath, cropParams):
    imageList = getImageList(inputPath)

    createOutputStructure(outputPath)

    for image in imageList:
        imageName = image

        if imageName.find('.jpg') > 0:
            imageName = createPngImage(image)

        cropImage(imageName, inputPath, os.path.join(outputPath, CANDIDATE_FOLDER), cropParams)

def start():
    args = defineParser().parse_args()

    cropParams = {
        "height": args.height/100,
        "width": args.width/100,
        "var": args.var/100,
        "aspect": readAspect(args.aspect),
        "crops": args.crops,
        "outputRes": readResolution(args.outputRes)
    }

    randomCrop(args.imagePath, args.out, cropParams)

start()


