#!/usr/bin/env python

import objectdataset as od
import tensorflow as tf
import os

import numpy as np
from traininghooks import TensorboardViewHook
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverListener

tf.logging.set_verbosity(tf.logging.INFO)

INPUT_SHAPE=[-1, 128, 96, 3]
L2_REGULARIZATION=0.2

def evalClassifier(object_classifier, eval_data, eval_labels):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=True)

    return object_classifier.evaluate(input_fn=eval_input_fn)

class CSVSaverListerner(CheckpointSaverListener):

    def __init__(self, additionalData, object_classifier,
                 train_data, train_labels,
                 eval_data, eval_labels ):
        self.data = additionalData
        self.classifier = object_classifier
        self.datasets = {
            "train": {
                "input": train_data,
                "labels": train_labels
            },
            "test": {
                "input": eval_data,
                "labels": eval_labels
            }
        }

        if not os.path.exists(additionalData['output']):
            os.makedirs(additionalData['output'])

        self.data['csvname'] = additionalData['output'] + '/trainingData.csv'
        self.prepared = False
        self.f = open(self.data['csvname'], 'a')

    def extractTensorValue(self, session, tensorName):
        ten = session.graph.get_tensor_by_name(tensorName)
        return session.run(ten)

    def saveToCSV(self, values):
        record = str(self.data['minibatch']) + ', ' + str(self.data['learning']) + ', ' + str(values['step']) + ', ' \
                 + str(self.data['dropout']) + ', ' + str(self.data['L2']) + ', ' \
                 + str(values['lossTest']) + ', ' + str(values['accuracyTest']) + ', ' \
                 + str(values['lossTrain']) + ', ' + str(values['accuracyTrain']) + '\n'

        self.f.write(record)
        self.f.flush()

    def after_save(self, session, global_step_value):
        accuracyTest = evalClassifier(self.classifier, self.datasets['test']['input'], self.datasets['test']['labels'])
        accuracyTrain = evalClassifier(self.classifier, self.datasets['train']['input'], self.datasets['train']['labels'])

        values = {
            "step": global_step_value,
            "lossTest": accuracyTest['loss'],
            "accuracyTest": accuracyTest['accuracy'],
            "lossTrain": accuracyTrain['loss'],
            "accuracyTrain": accuracyTrain['accuracy'],
        }

        self.saveToCSV(values)

    def end(self, session, global_step_value):
        self.f.close()

def conv_layer(name, input_layer, kernel, filters, l2):
    with tf.name_scope(name):
        conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=filters,
            kernel_size=kernel,
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2)
        )


        convSecond = tf.layers.conv2d(
            inputs=conv,
            filters=filters,
            kernel_size=kernel,
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2)
        )

        pool = tf.layers.max_pooling2d(inputs=convSecond, pool_size=[2, 2], strides=2)

    return pool

def createModelFn(learningRate, convLayers, dropout, l2):
    def cnn_model_fn(features, labels, mode):

        _, rows, columns, channels = INPUT_SHAPE

        # Define the input layer
        layers = [tf.reshape(features["x"], INPUT_SHAPE)]

        for index, kernel in enumerate(convLayers):
            filterSize, filterNumber = kernel
            currentLayer = conv_layer("Conv" + str(index + 1), layers[index], [filterSize, filterSize], filterNumber, l2)
            layers.append(currentLayer)


        # Flatten
        _, lastLayerRows, lastLayerColumns, lastLayerChannels = layers[-1].shape
        pool_flat = tf.reshape(layers[-1], [-1, lastLayerRows.value * lastLayerColumns.value * lastLayerChannels.value])

        with tf.name_scope('Dense'):
            # #1 Dense layer
            # Input Tensor Shape: [batch_size, 12288]
            # Output Tensor Shape: [batch_size, 1024]
            dropout1 = tf.layers.dropout( inputs=pool_flat, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

            dense1 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2))

            # Add dropout
            dropout2 = tf.layers.dropout( inputs=dense1, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

            # Logits layer
            # Input Tensor Shape: [batch_size, 1024]
            # Output Tensor Shape: [batch_size, 2]
            rawLogits = tf.layers.dense(inputs=dropout2, units=2, activation=tf.nn.sigmoid,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2))

            logits = tf.add(rawLogits, 1e-10)

        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            outputs = {"prediction": tf.estimator.export.PredictOutput(predictions)}

            return tf.estimator.EstimatorSpec(mode=mode, predictions=logits, export_outputs=outputs)

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.concat([1- labels, labels], 1)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=onehot_labels), name="reduced_loss")
        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss

        tf.summary.scalar("loss", loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdagradOptimizer(learning_rate=learningRate)
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

    return cnn_model_fn

def trainRecognizer(trainingData):
    # Load training and eval data
    dataset = od.loadImageSet(trainingData["input"],  15000)

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
        model_fn=createModelFn(trainingData['learning'], [[5, 64], [3, 128], [3, 256]],
                               trainingData['dropout'], trainingData['L2']),
        model_dir=trainingData['output'])

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Add Summary hook
    summary_hook = tf.train.SummarySaverHook(
        output_dir=trainingData['output'] + '/summary',
        save_steps=50,
        scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

    # Add information for the Tensorboard summary
    tensorboard_hook = TensorboardViewHook()

    listener = CSVSaverListerner(trainingData, object_classifier,
                                 train_data, train_labels,
                                 eval_data, eval_labels)

    saver_hook = tf.train.CheckpointSaverHook(
        trainingData['output'],
        listeners=[listener],
        save_steps=500)


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=trainingData["minibatch"],
        num_epochs=None,
        shuffle=True)

    object_classifier.train(
        input_fn=train_input_fn,
        steps=trainingData["iterations"],
        hooks=[logging_hook, saver_hook, summary_hook])

    # Evaluate the model and print results
    eval_results = evalClassifier(object_classifier, eval_data, eval_labels)

    def serving_input_receiver_fn():
        feature_spec = {'x': tf.FixedLenFeature(INPUT_SHAPE[1:4], tf.float32)}
        serialized_tf_example = tf.placeholder(dtype=tf.float32,
                                               shape=INPUT_SHAPE[1:4],
                                               name='input_tensors')

        receiver_tensors = {'inputs': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    placeholderInput = tf.placeholder(dtype=tf.float32,
                                            shape=INPUT_SHAPE[1:4],
                                            name='input_tensors')

    alternative_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({"x": placeholderInput})
    object_classifier.export_savedmodel(trainingData['output'], alternative_fn)


def parseArguments(args):
    if (len(args) != 8):
        print("Wrong number of parameters. Defaulting to internal values.")
        return {
            "input": './results/augmented',
            "output": '/tmp/object_convnet1',
            "L2": 0.1,
            "dropout": 0.5,
            "iterations": 1000,
            "minibatch": 100,
            "learning": 1e-3
        }
    else:
        return {
            "input": args[1],
            "output": args[2],
            "L2": float(args[3]),
            "dropout": float(args[4]),
            "iterations": float(args[5]),
            "minibatch": int(args[6]),
            "learning": float(args[7])
        }

def main(argv):
    args = parseArguments(argv)
    trainRecognizer(args)

if __name__ == "__main__":
    tf.app.run()
