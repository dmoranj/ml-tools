#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

INPUTMODEL='./results/object_convnet1/1515141257'

def getInputData(imagePath):
    image = mpimg.imread(imagePath)
    normalized = np.asarray(image[:, :, :3], dtype=np.float32)
    return normalized


with tf.Session(graph=tf.Graph()) as sess:
    metagraphdef = tf.saved_model.loader.load(sess, ['serve'], INPUTMODEL)

    input = tf.get_default_graph().get_tensor_by_name('input_tensors:0')
    output = tf.get_default_graph().get_tensor_by_name('softmax_tensor:0')

    input_data = getInputData('./results/augmented/Negative/2_5_2015-09-26 15.32.21.png')

    preds = sess.run(output, feed_dict={"input_tensors:0": input_data})
    print('Probabilites: ' + str(preds))


