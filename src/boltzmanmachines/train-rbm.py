import binaryImages as bm
import imageUtils as iu
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
import os


INPUT_FOLDER='../../ill-examples'


def contrastiveDivergenceModel(input, cd, nh, epsilon):
    m, nv = input.shape

    visible, b, cd0, cdN, hidden, reconstruction, weights = createVariables(nh, nv)

    reconstruction.assign(visible)

    updateHidden(b, hidden, nh, reconstruction, weights)
    updateReconstruction(hidden, nv, reconstruction, weights)

    cd0.assign_add(contrastiveDivergence(hidden, reconstruction)/m)

    for i in range(0, cd):
        updateHidden(b, hidden, nh, reconstruction, weights)
        updateReconstruction(hidden, nv, reconstruction, weights)

    cdN.assign_add(contrastiveDivergence(hidden, reconstruction)/m)

    dW = epsilon*(cd0 - cdN)

    return visible, weights, dW


def weightUpdateModel(weight, dW):
    return tf.assign(weight, tf.add(weight, dW))


def createVariables(nh, nv):
    visible = tf.placeholder(tf.float32, [nv, 1])
    weights = tf.Variable(tf.random_uniform([nv, nh]), name="weights")
    reconstruction = tf.Variable(tf.zeros(visible.shape), name="reconstruction")
    hidden = tf.Variable(tf.zeros((nh, 1)))
    b = tf.Variable(tf.zeros([nh, 1]), name="biasHidden")
    cd0 = tf.Variable(tf.zeros(weights.shape))
    cdN = tf.Variable(tf.zeros(weights.shape))
    return visible, b, cd0, cdN, hidden, reconstruction, weights


def contrastiveDivergence(hidden, reconstruction):
    return tf.matmul(reconstruction, tf.transpose(hidden))


def updateReconstruction(hidden, nv, reconstruction, weights):
    # Negative Phase (clamped hidden)
    # Calculate probabilities of being on for each visible unit
    E = tf.matmul(weights, hidden)
    p = tf.sigmoid(-E)
    # Draw the correspondent input based on the probabilities
    return tf.assign(reconstruction, tf.to_float(tf.less_equal(tf.random_uniform((nv, 1)), p)))


def updateHidden(b, hidden, nh, reconstruction, weights):
    # Positive Phase (clamped input)
    # Calculate probabilities of being on for each hidden unit
    E = tf.add(tf.matmul(weights, reconstruction, transpose_a=True), b)
    p = tf.sigmoid(-E)
    # Draw the correspondent state based on the probabilities
    return tf.assign(hidden, tf.to_float(tf.less_equal(tf.random_uniform((nh, 1)), p)))


def tryImage(imagePath):
    image = iu.loadPngImage(imagePath)
    representation = bm.getBinaryImage(image)
    recovered = bm.recoverImage(representation, image.shape)
    plt.imshow(image)
    plt.show()


def trainRBM(input, nh, epochs, session):
    m, _ = input.shape

    # Create CD Model
    visible, weights, dW = contrastiveDivergenceModel(input, 1, nh, 0.3)

    # Create Weight Update model
    weightUpdate = weightUpdateModel(weights, dW)

    # Initialize variables
    init_op = tf.global_variables_initializer()
    session.run(init_op)

    # Execute for N epochs
    for i in range(0, epochs):
        # For each epoch, show all the inputs and calculate the average dW
        for j in range(0, m):
            example = input[j, :]
            newInput = np.reshape(example, (example.shape[0], 1))

            session.run([dW], {visible: newInput})

        # Update the weights
        session.run(weightUpdate)

    # Return the weights
    return weights.eval()


def loadInput(inputPath):
    imageList = glob.glob(os.path.join(inputPath, "*.png"))

    results = []

    for imagePath in imageList:
        image = iu.loadPngImage(imagePath)
        representation = bm.getBinaryImage(image)
        results.append(representation)

    return np.array(results)


def executeTraining():
    # Load input
    input = loadInput(INPUT_FOLDER)

    # Train RBM
    with tf.Session() as session:
        writer = tf.summary.FileWriter("/tmp/boltzman", session.graph)

        weights = trainRBM(input, 500, 1, session)

        # Sample RBM
        print('RBM Trained')

        writer.close()


executeTraining()

