# Machine Learning Tools

## Introduction

This repository holds utilities for creation and experimentation with new datasets,
dataset augmentation, preprocessing and other tasks related with machine learning. All 
these utilities are still under development, so they are best used as a source for 
experimentation.

## Dataset creation

The `/data_augmentation` foder contains a set of tools oriented to the creation of new
datasets based on sets of images. Here is a brief description of their use:
 
 
 
This set of tools was designed to be used in cascade, to crop image section, identify 
those sections that meet our needs, and augment the cleanead data. It works with a
default directory structure, that contains the following subfolders (the default
root folder is `./results` under the tool execution path):

* `./candidates`: this folder will hold all the image crops obtained from the original images
* `./augmented`: this folder will typically hold the sets of images that resulted from the data-augmentation process.
* `./selection`:
* `./data`:


The following sections expand the information about each one of the tools. Command customizing options can be found 
by executing each script with the `--help` option. 

### Random cropping (random-cropping.py)

The purpose of these tool is to process all the candidate images for a dataset, generating, for each one of them
a set of randomly cropped subimages, expecting to capture interesting objects to be labeled in the process.
        
The tool lets the user tune the general parameters of the random cropping, so to maximize the possibility
of generating cropped images that are interesting for the task at hand. It will also help in making the
dataset homogeneous, by down or upsizing each of the cropped segments to a predetermined size if required.
        
The cropping tool will generate an output subfolder "/candidates" containing all the cropped images.
        
If the --aspect option is used, the height parameter will be ignored, and the frame height will be adapted
to the calculated width of the image.


### Data augmentation (data-augmentation.py)

This tool performs data augmentation on all the images in the provided dataset folder. The data augmentation 
techniques used are the following (the number in parenthesis indicates how many times the transformation is
randomly applied):
        
* Horizontal image mirroring (1) 
* Slight rotations (2)
* Tone modifications (5)

The modifications are applied in a cascade-style (i.e.: each type of modification is applied to each of the
images that resulted from the previous one) so the total number of new images per original image is 20. The
set of transformations can be easily edited in the script by editing the `rotations` and `tones` arrays in the
`dataAugmentImage()` function.

The data-agumentation process will perform recursively in the provided directory, and the generated results
folder will replicate the structure of the original (i.e.: if the original folder contains a dataset divided
into *training*, *testing* and *cv* folders, all of them would be separatedly augmented, and the structure preserved 
in the target folder).

All the unmodified images will be also copied to the destination location.

### Generate boxing (generate-boxing.py)


## Boltzmann machines (/boltzmanmachines)

This folder contains some preliminary experiments with Restricted Boltzman Machines (RBMs).
Currently, there is just one experiment, the *train-rbm.py* script, that trains a RBM
with a single layer of input nodes and a single layer of hidden units, using Contrastive
Divergence to store and recognize a set of images. All the algorithm is directly
implemented using low level Tensorflow units. Binary input units are used to codify the
image, so the algorithm is very memory intensive (it may easily give a memory error for
big enough images).

This script is in a very early stage and doesn't have a command line interface yet. All 
its parameters are hardcoded inside the code. 

There are two important constants to edit for experimentation with the script:

* INPUT_FOLDER: this constant should have the path of the folder containing the examples
 the RBM is supposed to learn.

* ITERATIONS_TO_STATIONARY: number of iterations of unit value updating before considering 
the distribution is stationary.

When the script is executed without parameters it will:
* Load all the images in the examples folder.
* Turn the images into binary Numpy arrays.
* Train the RBM with the images.



