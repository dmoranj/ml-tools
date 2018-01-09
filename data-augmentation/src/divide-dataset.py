import argparse
import os
import shutil
from fileutils import getSubfolders
import glob

DEFAULT_OUTPUT_PATH = "./results/datasets"


def generateDescription():
    return """
        Divide all the images in a directory into n different buckets, for its use as training, cross-validation and
        testing (for one or multiple estimators). 
        
        The execution generates a new output folder ("./results/datasets" by default) that will contain one folder
        per bucket. Each bucket will contain the corresponding percentages of examples, structured in the same way
        they were structured in the original folder
    """


def defineParser():
    parser = argparse.ArgumentParser(description=generateDescription())

    parser.add_argument('inputFolder', type=str, help='Original folder containing all the examples')
    parser.add_argument('buckets', type=str, help='String containing a comma separated list of bucket names')
    parser.add_argument('ratios', type=str, help='String containing a comma separated list of ratios')

    return parser


def createFolders(output, buckets, subfolders):
    if not os.path.exists(output):
        os.mkdir(output)

    for bucket in buckets:
        bucketFolder = os.path.join(output, bucket)

        if not os.path.exists(bucketFolder):
            os.mkdir(bucketFolder)

        for subfolder in subfolders:
            bucketSubfolder = os.path.join(bucketFolder, subfolder)

            if not os.path.exists(bucketSubfolder):
                os.mkdir(bucketSubfolder)


def listImagesByFolder(input, subfolders):

    imageList = []

    for subfolder in subfolders:
        subfolderPath = os.path.join(input, subfolder)
        listPerExtension = [glob.glob(os.path.join(subfolderPath, extension)) for extension in ["*.png", "*.jpg"]]
        imageList.append([f for sub in listPerExtension for f in sub])

    return imageList


def divideImagesByRatios(dataset, ratios):

    results = []

    for subfolderData in dataset:
        rows = len(subfolderData)
        lastIndex = 0
        subfolderResults = []

        for ratio in ratios:
            nextIndex = int(min(lastIndex + ratio*rows, rows))
            sliced = subfolderData[lastIndex:nextIndex]
            subfolderResults.append(sliced)
            lastIndex = nextIndex

        results.append(subfolderResults)

    return results


def copyImagesToDatasets(imageDataset, output, buckets, subfolders):

    for keySubfolder, subfolder in enumerate(imageDataset):
        for keyBucket, bucket in enumerate(subfolder):
            subfolderPath = os.path.join(output, buckets[keyBucket], subfolders[keySubfolder])
            targetImagesNames = [os.path.join(subfolderPath, imageName.split("/")[-1]) for imageName in bucket]

            for idx, originalImage in enumerate(bucket):
                shutil.copy(originalImage, targetImagesNames[idx])


def divideDataset(options):
    print("Dividing dataset")

    subfolders = getSubfolders(options['input'])[1:]
    createFolders(options['output'], options['buckets'], subfolders)
    imageDataset = listImagesByFolder(options['input'], subfolders)
    imageDivision = divideImagesByRatios(imageDataset, options['ratios'])
    copyImagesToDatasets(imageDivision, options['output'], options['buckets'], subfolders)


def parseArgs(args):
    options = {}

    options['input'] = args.inputFolder
    options['ratios'] = [float(s.strip()) for s in args.ratios.split(",")]
    options['buckets'] = [s.strip() for s in args.buckets.split(",")]
    options['output'] = DEFAULT_OUTPUT_PATH

    return options


def start():
    args = defineParser().parse_args()

    options = parseArgs(args)

    divideDataset(options)


start()
