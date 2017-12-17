import objectdataset as od
import tensorflow as tf

import numpy as np
from traininghooks import variable_summaries
from traininghooks import TensorboardViewHook
#from tf.train import CheckpointSaverListener

tf.logging.set_verbosity(tf.logging.INFO)

# class CSVSaverListerner(CheckpointSaverListener):
#     def begin(self):
#         # You can add ops to the graph here.
#         print('Starting the session.')
#         self.your_tensor = ...
#
#     def before_save(self, session, global_step_value):
#         print('About to write a checkpoint')
#
#     def after_save(self, session, global_step_value):
#         print('Done writing checkpoint.')
#
#     def end(self, session, global_step_value):
#         print('Done with the session.')



def conv_layer(name, input_layer, kernel, filters):
    with tf.name_scope(name):
        # #1 convolutional layer
        # Input Tensor Shape: [batch_size, 128, 96, 3]
        # Output Tensor Shape: [batch_size, 128, 96, 32]
        conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=filters,
            kernel_size=kernel,
            padding="same",
            activation=tf.nn.relu)

        # #1 pooling layer
        # Input Tensor Shape: [batch_size, 128, 96, 32]
        # Output Tensor Shape: [batch_size, 64, 48, 32]
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

    return pool

def cnn_model_fn(features, labels, mode):
    # Define the input layer
    input_layer = tf.reshape(features["x"], [-1, 64, 48, 3])

    # Input Tensor Shape: [batch_size, 64, 48, 3]
    # Output Tensor Shape: [batch_size, 32, 24, 32]
    conv1 = conv_layer("Conv1", input_layer, [3, 3], 32)

    # Input Tensor Shape: [batch_size, 32, 24, 32]
    # Output Tensor Shape: [batch_size, 16, 12, 64]
    conv2 = conv_layer("Conv2", conv1, [3, 3], 64)

    # Input Tensor Shape: [batch_size, 16, 12, 64]
    # Output Tensor Shape: [batch_size, 8, 6, 128]
    conv3 = conv_layer("Conv3", conv2, [3, 3], 128)

    # Flatten
    # Input Tensor Shape: [batch_size, 8, 6, 128]
    # Output Tensor Shape: [batch_size, 3072]
    pool_flat = tf.reshape(conv3, [-1, 8 * 6 * 128])

    with tf.name_scope('Dense'):
        # #1 Dense layer
        # Input Tensor Shape: [batch_size, 6144]
        # Output Tensor Shape: [batch_size, 512]
        dense1 = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)

        # Add dropout
        #dropout = tf.layers.dropout(
        #    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, 2]
        rawLogits = tf.layers.dense(inputs=dense1, units=2, activation=tf.nn.sigmoid)
        logits = tf.add(rawLogits, 1e-8)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.concat([1- labels, labels], 1)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=onehot_labels))

    tf.summary.scalar("loss", loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=1e-3)
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

    train_data = np.asarray(dataset['train']['images'], dtype=np.float32)
    train_labels = np.asarray(dataset['train']['labels'], dtype=np.float32)
    eval_data = np.asarray(dataset['test']['images'], dtype=np.float32)
    eval_labels = np.asarray(dataset['test']['labels'], dtype=np.float32)

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

    # Add Checkpoint hooks
    checkpoint_hook = tf.train.CheckpointSaverHook("/tmp/object_convnet_1", save_steps=50)

    # Add Summary hook
    summary_hook = tf.train.SummarySaverHook(
        output_dir='/tmp/object_convnet_1/summary',
        save_steps=50,
        scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

    # Add information for the Tensorboard summary
    tensorboard_hook = TensorboardViewHook()

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
        hooks=[logging_hook, checkpoint_hook, summary_hook])

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
