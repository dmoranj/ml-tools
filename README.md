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

* `./candidates`: holds all the image crops obtained from the original images
* `./augmented`: holds the sets of images that resulted from the data-augmentation process.
* `./selection`: this is the destination of the boxings and highlighting from the boxing generation script.
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

### Object recognition (objectrecognizer.py and metaparameter-search.py)

These scripts contain an implementation of an object recognizer based on the typical CNN+FC architecture. The object
recognition script is implemented with the TensorFlow Estimator API, using Adagrad as the optimizer. The arquitechture
of the network can be configured inside the script, by using a string like the following:

```buildoutcfg
[[5, 64, 2], [3, 128, 3], [3, 128, 3], [3, 256, 3]]
```

Each of the subarrays indicates a new Network Layer composed of a number of CNN layers and its correspondent final 
max pooling layer. The first number in the Network Layer definition indicates the size of the convolution kernel, the 
second one the number of filters, and the third the number of convolutional layers before the max pooling.  

The `objectrecognizer.py` script receives the following arguments:

* **input**: folder containing the image examples.
* **output**: folder that will hold the trained model (or the model to continue training).
* **L2**: L2 regularizer constant [0..1].
* **dropout**: dropout level [0..1].
* **iterations**: number of iterations to train for.
* **minibatch**: minibatch size.
* **learning**: learning rate.

The Estimator API takes care of automatically saving and restoring training given an input folder. Statistics of the
training can be found in th `training.csv` file inside the model folder.

The `metaparameter-search.py` script was conceived to aid in searching for the most appropriate metaparameters for the
object-recognition algorithm. It generates random sets of parameters normally distributed about a given mean, with a
specified variance, with values limited to a certain interval (function `generateValues()`). The object recognizer
script is run against all the examples in this loose grid for a certain number of iterations, saving the results
in the CSV files. R Markdown file `analyzeTraining.Rmd` contains some functions that may help in analyzing the
results.



### Generate boxing (generate-boxing.py)

This tool generates a random boxing pattern for a set of images, running an object recognizing algorithm
for each patch corresponding to a box. The object recognition algorithm is supposed to have been trained with
this same toolset `objectrecognizer.py` script (the reason for that is that input and output tensors for prediction 
are retrieved by name; a small change in the script should make it usable with different models). 

The boxing division of the image can be used in two tasks:
        
* Creation of datasets of images labeled with object location and boxing.
* Creation of image patches divided according to the algorithm classification.
* Highlight masks for the selected objects, based on box superposition
        
The second task is mainly thought as a mean to extend the original object recognition dataset, by manually choosing, 
from the division performed by the algorithm, the false positives and false negatives, thus making interesting
cases that may improve the algorithm's performance in real data.

The highlighting algorithm works by using the generated random boxes to darken or lighting the corresponding image 
patch, depending on whether the patch corresponded to a positive example or a negative one. The highlighting procedure
performance depends heavily on:

* the size of the boxes relative to the object to be recognize.
* the number of iterations the algorithm is executed.
* and the performance of the classifier itself.

Highlighted images are stored in the `./selection` subfolder.

Dataset boxing creation doesn't currently save the generated boxing, but it displays it overimposed to the shown
images.


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



