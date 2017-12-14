import os

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

