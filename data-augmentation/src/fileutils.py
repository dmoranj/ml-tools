import os

def createName(image, outputPath, index):
    nameParts = image.split("/")
    return os.path.join(outputPath, str(index) + "_" + nameParts[-1])


