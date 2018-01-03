#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci


INPUTMODEL='./results/object_convnet1/model.ckpt-11200.meta'
LATESTCHECKPOINT='./results/object_convnet1/'


def visualizeWeights(weights):
    print('Visualizing weights')
    print('- Conv0 size = ' + str(weights['conv0'][0].shape))
    print('- Conv1 size = ' + str(weights['conv1'][0].shape))
    print('- Conv2 size = ' + str(weights['conv2'][0].shape))
    print('- Conv3 size = ' + str(weights['conv3'][0].shape))

    conv1img = np.zeros((49, 49, 3))
    conv1img.fill(-10)

    for i in range(0, 8):
        for j in range(0, 8):
            index = i*8 + j
            kernel = weights['conv0'][0][:, :, :, index]

            initX = i*5 + i +1
            initY = j*5 + j +1

            endX = initX + 5
            endY = initY + 5

            conv1img[initX:endX, initY:endY, :] = kernel

    plt.imshow(sci.special.expit(10*conv1img))
    plt.show()


def main():
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(INPUTMODEL)

    with tf.Session() as sess:
        imported_meta.restore(sess, tf.train.latest_checkpoint(LATESTCHECKPOINT))
        print('Loaded')

        weights = { 'conv0': sess.run([tf.get_default_graph().get_tensor_by_name('conv2d/kernel:0')]),
                    'conv1': sess.run([tf.get_default_graph().get_tensor_by_name('conv2d_1/kernel:0')]),
                    'conv2': sess.run([tf.get_default_graph().get_tensor_by_name('conv2d_2/kernel:0')]),
                    'conv3': sess.run([tf.get_default_graph().get_tensor_by_name('conv2d_3/kernel:0')])
        }

        visualizeWeights(weights)


main()