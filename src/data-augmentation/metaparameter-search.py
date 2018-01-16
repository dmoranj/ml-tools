#!/usr/bin/env python

import objectrecognizer as obrec
import numpy as np

DEFAULT_ARGS = {
     "input": './results/augmented',
     "output": './results/tentative_convnet',
     "iterations": 15000,
}

globalId = 11

def generateArgs(learning, minibatch, dropout, l2):
    global globalId

    args = DEFAULT_ARGS.copy()

    args['minibatch'] = minibatch
    args['learning'] = learning
    args['L2'] = l2
    args['dropout'] = dropout
    args['output'] = args['output'] + '-' + str(globalId)

    globalId += 1

    return args

def executeExperiments(experimentValues):
    for experiment in experimentValues:
        print('\n\nTraining for learning rate {} minibatch {} dropout {} and L2 {}\n'.format(
            experiment[0], experiment[1], experiment[2], experiment[3]))
        args = generateArgs(experiment[0], experiment[1], experiment[2], experiment[3])
        obrec.trainRecognizer(args)


def generateValues(n, avg, var,  minimum=1e-4, maximum=0.1, typeFunction = float):
    rawValues = np.random.normal(float(avg), float(var), n)
    values = np.maximum(np.minimum(rawValues, maximum), minimum)
    return map(typeFunction, values)


executeExperiments(list(zip(
    generateValues(5, 0.002, 0.002),
    generateValues(5, 120, 20, 50, 200, int),
    generateValues(5, 0.3, 0.2, 0.1, 0.5),
    generateValues(5, 0.2, 0.15, 0.1, 0.5)
)))
