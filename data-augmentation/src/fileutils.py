import os
import re
import glob
import uuid

def createName(image, outputPath, index):
    nameParts = image.split("/")
    return os.path.join(outputPath, str(index) + "_" + nameParts[-1])

def readAspect(aspect):
    if aspect:
        components = [int(x) for x in aspect.split(":")]
        ratio = float(components[1])/float(components[0])
        return ratio
    else:
        return None

def readResolution(resolution):
    if resolution:
        components = [int(x) for x in resolution.split("x")]
        return components
    else:
        return None


def getSubfolders(path):
    return [re.sub(r"^/", '', x[0].replace(path, '')) for x in os.walk(path)]


def getImageList(folder, extension="*.png"):
    return glob.glob(os.path.join(folder, extension))

def generateName():
    return str(uuid.uuid1()) + "_"

