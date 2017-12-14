
import argparse

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
    """
def defineParser():
    parser = argparse.ArgumentParser(description=generateDescription())
    parser.add_argument('imagePath', type=str, help='Path to the data directory')
    parser.add_argument('--outputRes', dest='outputRes', type=str,
                        help='Output resolution of the cropped images')

    return parser

def dataAugment(imagePath, augmentOptions):
    print("Augmenting data")

def start():
    args = defineParser().parse_args()
    dataAugment(args.imagePath, args)

