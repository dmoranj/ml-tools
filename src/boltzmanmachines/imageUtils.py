from PIL import Image, ExifTags
import numpy as np
import matplotlib.image as mpimg

def transformFromJpg(image, exif):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        if exif == None:
            return image

        exif = dict(image._getexif())

        if exif[orientation] == 3:
            im = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            im = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            im = image.rotate(90, expand=True)
        else:
            im = image

        return im
    except (AttributeError, KeyError, IndexError):
        print('EXIF information not found for "' + image + '"')
        return image


def createPngImage(image):
    im = Image.open(image)
    im = transformFromJpg(im)

    imageName = image.replace("jpg", "png")
    im.save(imageName, "PNG")
    im.close()

    return imageName


def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def loadJpegImage(imagePath):
    image = Image.open(imagePath)

    if len(image.getbands()) < 3:
        image = image.convert('RGB')

    if hasattr(image, '_getexif'):
        exif = image._getexif()
    else:
        exif = None

    image = PIL2array(transformFromJpg(image, exif))

    normalized = np.asarray(image[:, :, :3], dtype=np.float32)/255
    return normalized

def loadPngImage(imagePath):
    image = mpimg.imread(imagePath)
    return image