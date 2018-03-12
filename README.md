# Machine Learning Tools

## Introduction

This repository holds utilities for creation and experimentation with new datasets,
dataset augmentation, preprocessing and other tasks related with machine learning.

## Dataset creation



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



