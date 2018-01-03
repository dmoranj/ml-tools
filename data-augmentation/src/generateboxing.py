#!/usr/bin/env python

import tensorflow as tf

INPUTMODEL='./results/object_convnet1/model.ckpt-11200.meta'
LATESTCHECKPOINT='./results/object_convnet1/'


#with tf.Session(graph=tf.Graph()) as sess:
#    tf.saved_model.loader.load(sess, [tag_constants.TRAINING], INPUTMODEL)

tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph(INPUTMODEL)

with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint(LATESTCHECKPOINT))
    print('Loaded')

    print(sess.run([tf.get_default_graph().get_tensor_by_name('conv2d/kernel:0')]))
