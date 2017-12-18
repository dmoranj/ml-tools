#!/usr/bin/env python

import objectrecognizer as obrec

DEFAULT_ARGS = {
     "input": './results/augmented',
     "output": '/tmp/object_convnet',
     "testTrainBalance": 0.85,
     "iterations": 20000,
}

def generateArgs(learning, minibatch, index1, index2):
    args = DEFAULT_ARGS.copy()

    args['minibatch'] = minibatch
    args['learning'] = learning
    args['output'] = args['output'] + str(index1) + '_' + str(index2)

    return args

def executeExperiments():
    learningValues = [1e-3, 5e-4, 1e-4]
    minibatchValues = [25, 50, 100, 200]

    for i, learning in enumerate(learningValues):
        for j, minibatch in enumerate(minibatchValues):
            print('\n\nTraining for learning rate {} and minibatch {}\n'.format(learning, minibatch))
            args = generateArgs(learning, minibatch, i, j)
            obrec.trainRecognizer(args)


executeExperiments()