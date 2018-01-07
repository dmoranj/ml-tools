#!/usr/bin/env python

import objectrecognizer as obrec

DEFAULT_ARGS = {
     "input": './results/augmented',
     "output": '/tmp/object_convnet5L',
     "testTrainBalance": 0.85,
     "iterations": 20000,
}

def generateArgs(learning, minibatch, index1, index2):
    args = DEFAULT_ARGS.copy()

    args['minibatch'] = minibatch
    args['learning'] = learning
    args['output'] = args['output'] + str(index1) + '_' + str(index2)

    return args

def executeExperiments(learningValues, minibatchValues):
    for i, learning in enumerate(learningValues):
        for j, minibatch in enumerate(minibatchValues):
            print('\n\nTraining for learning rate {} and minibatch {}\n'.format(learning, minibatch))
            args = generateArgs(learning, minibatch, i, j)
            obrec.trainRecognizer(args)


executeExperiments([7e-3, 5e-3, 2e-3], [20, 35, 50])