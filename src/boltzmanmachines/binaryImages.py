import numpy as np

def loadImage(image):
    intImage = (image*255).astype(np.uint8)
    representation = np.unpackbits(intImage)
    return representation

def recoverImage(representation, shape):
    intArray = np.packbits(representation)
    image = np.reshape(intArray, shape)/255

    return image

