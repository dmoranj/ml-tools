import objectdataset as od
import tensorflow as tf

import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Define the input layer
    input_layer = tf.reshape(features["x"], [-1, 128, 96, 3])

    with tf.name_scope('Conv1'):
        # #1 convolutional layer
        # Input Tensor Shape: [batch_size, 128, 96, 3]
        # Output Tensor Shape: [batch_size, 128, 96, 32]
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # #1 pooling layer
        # Input Tensor Shape: [batch_size, 128, 96, 32]
        # Output Tensor Shape: [batch_size, 64, 48, 32]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    with tf.name_scope('Conv2'):
        # #2 Convolutional layer
        # Input Tensor Shape: [batch_size, 64, 48, 32]
        # Output Tensor Shape: [batch_size, 64, 48, 64]
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

        # #2 pooling layer
        # Input Tensor Shape: [batch_size, 64, 48, 64]
        # Output Tensor Shape: [batch_size, 32, 24, 64]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    with tf.name_scope('Conv3'):
        # #3 Convolutional layer
        # Input Tensor Shape: [batch_size, 32, 24, 64]
        # Output Tensor Shape: [batch_size, 32, 24, 128]
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

        # #3 pooling layer
        # Input Tensor Shape: [batch_size, 32, 24, 128]
        # Output Tensor Shape: [batch_size, 16, 12, 128]
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    with tf.name_scope('Conv4'):
        # #4 Convolutional layer
        # Input Tensor Shape: [batch_size, 16, 12, 128]
        # Output Tensor Shape: [batch_size, 16, 12, 256]
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

        # #4 pooling layer
        # Input Tensor Shape: [batch_size, 16, 12, 256]
        # Output Tensor Shape: [batch_size, 8, 6, 256]
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Flatten
    # Input Tensor Shape: [batch_size, 8, 6, 96]
    # Output Tensor Shape: [batch_size, 12288]
    pool4_flat = tf.reshape(pool4, [-1, 8 * 6 * 256])

    with tf.name_scope('Dense'):
        # #1 Dense layer
        # Input Tensor Shape: [batch_size, 12288]
        # Output Tensor Shape: [batch_size, 1024]
        dense1 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu)

        # Add dropout
        #dropout = tf.layers.dropout(
        #    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits layer
        # Input Tensor Shape: [batch_size, 512]
        # Output Tensor Shape: [batch_size, 2]
        rawLogits = tf.layers.dense(inputs=dense2, units=2)
        logits = tf.add(rawLogits, 1e-9)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.concat([1 - labels, labels], 1)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    dataset = od.loadImageSet('./results/augmented', 0.8)

    train_data = np.asarray(dataset['train']['images'], dtype=np.float16)
    train_labels = np.asarray(dataset['train']['labels'], dtype=np.int32)
    eval_data = np.asarray(dataset['test']['images'], dtype=np.float16)
    eval_labels = np.asarray(dataset['test']['labels'], dtype=np.int32)

    assert(not np.isnan(train_data).any())
    assert(not np.isnan(train_labels).any())
    assert(not np.isnan(eval_data).any())
    assert(not np.isnan(eval_labels).any())

    # Create the Estimator
    object_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/object_convnet_1")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=50,
        num_epochs=None,
        shuffle=True)

    object_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = object_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
