import binaryImages as bm
import imageUtils as iu
import matplotlib.pyplot as plt

INPUT_FOLDER='../../ill-examples'

def tryImage(imagePath):
    image = iu.loadPngImage(imagePath)
    representation = bm.loadImage(image)
    recovered = bm.recoverImage(representation, image.shape)
    plt.imshow(image)
    plt.show()



