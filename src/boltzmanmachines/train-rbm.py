import binaryImages as bm
import imageUtils as iu
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
import os


INPUT_FOLDER='../../ill-examples'
ITERATIONS_TO_STATIONARY=20

def contrastiveDivergenceModel(input, cd, nh, epsilon):
    m, nv = input.shape

    with tf.name_scope('Training'):
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
    visible = tf.placeholder(tf.float16, [nv, 1])
    weights = tf.Variable(tf.random_uniform([nv, nh], dtype=tf.float16), dtype=tf.float16, name="weights")
    reconstruction = tf.Variable(tf.zeros(visible.shape, dtype=tf.float16), dtype=tf.float16, name="reconstruction")
    hidden = tf.Variable(tf.zeros((nh, 1), dtype=tf.float16), dtype=tf.float16)
    b = tf.Variable(tf.zeros([nh, 1], dtype=tf.float16), dtype=tf.float16, name="biasHidden")
    cd0 = tf.Variable(tf.zeros(weights.shape, dtype=tf.float16), dtype=tf.float16)
    cdN = tf.Variable(tf.zeros(weights.shape, dtype=tf.float16), dtype=tf.float16)
    return visible, b, cd0, cdN, hidden, reconstruction, weights


def contrastiveDivergence(hidden, reconstruction):
    return tf.matmul(reconstruction, tf.transpose(hidden))


def updateReconstruction(hidden, nv, reconstruction, weights):
    # Negative Phase (clamped hidden)
    # Calculate probabilities of being on for each visible unit
    E = tf.matmul(weights, hidden)
    p = tf.sigmoid(E)
    # Draw the correspondent input based on the probabilities
    return tf.assign(reconstruction, tf.cast(tf.less_equal(tf.random_uniform((nv, 1), dtype=tf.float16), p), tf.float16))


def updateHidden(b, hidden, nh, reconstruction, weights):
    # Positive Phase (clamped input)
    # Calculate probabilities of being on for each hidden unit
    E = tf.add(tf.matmul(weights, reconstruction, transpose_a=True), b)
    p = tf.sigmoid(E)
    # Draw the correspondent state based on the probabilities
    return tf.assign(hidden, tf.cast(tf.less_equal(tf.random_uniform((nh, 1), dtype=tf.float16), p), tf.float16))


def tryImage(imagePath):
    image = iu.loadPngImage(imagePath)
    representation = bm.getBinaryImage(image)
    recovered = bm.recoverImage(representation, image.shape)
    plt.imshow(image)
    plt.show()


def trainRBM(input, nh, epochs, session):
    m, _ = input.shape

    # Create CD Model
    visible, weights, dW = contrastiveDivergenceModel(input, 2, nh, 0.5)

    # Create Weight Update model
    weightUpdate = weightUpdateModel(weights, dW)

    # Initialize variables
    init_op = tf.global_variables_initializer()
    session.run(init_op)

    # Execute for N epochs
    for i in range(0, epochs):
        if i % 100 == 0:
            print("Training epoch N. " + str(i))

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
        shape = image.shape
        representation = bm.getBinaryImage(image)
        results.append(representation)

    return shape, np.array(results)


def createSamplingVariables(nh, nv):
    weights = tf.placeholder(shape=[nv, nh], dtype=tf.float16, name="trainedWeights")
    reconstruction = tf.Variable(tf.zeros([nv, 1], dtype=tf.float16), dtype=tf.float16, name="reconstruction")
    hidden = tf.Variable(tf.zeros((nh, 1), dtype=tf.float16), dtype=tf.float16)
    b = tf.Variable(tf.zeros([nh, 1], dtype=tf.float16), dtype=tf.float16, name="biasHidden")
    return  b, hidden, reconstruction, weights


def createSamplingModel(nh, nv):
    with tf.name_scope('Sampling'):
        b, hidden, reconstruction, weights = createSamplingVariables(nh, nv)

        reconstruction = reconstruction.assign(tf.random_uniform([nv, 1], dtype=tf.float16))

        for i in range(0, ITERATIONS_TO_STATIONARY):
            reconstruction = updateReconstruction(hidden, nv, reconstruction, weights)
            hidden = updateHidden(b, hidden, nh, reconstruction, weights)

        return weights, reconstruction


def sampleRBM(trainedWeights, session):
    nv, nh = trainedWeights.shape

    weights, reconstruction = createSamplingModel(nh, nv)

    init_op = tf.global_variables_initializer()
    session.run(init_op)

    generatedImage = session.run(reconstruction, {weights: trainedWeights})
    return generatedImage.astype(np.uint8)


def showResults(recovered, weights):

    _, nh = weights.shape

    plt.imshow(recovered)
    plt.show()

    fig=plt.figure(figsize=(8, 8))
    indices =  np.random.randint(0, nh, size=8*8)
    rows, columns = 8, 8
    idx = 1

    for i in indices:
        fig.add_subplot(rows, columns, idx)
        row = weights[:, i]
        image = bm.recoverImage(row > 0.5, recovered.shape)
        plt.imshow(image)
        idx = idx + 1

    plt.show()

def executeTraining():
    # Load input
    shape, input = loadInput(INPUT_FOLDER)

    # Train RBM
    with tf.Session() as session:
        writer = tf.summary.FileWriter("/tmp/boltzman", session.graph)

        weights = trainRBM(input, 100, 20000, session)
        print("Weights created with shape: " + str(weights.shape))
        writer.close()


    with tf.Session() as session:
        # Sample RBM
        sampledRepresentation = sampleRBM(weights, session)

        print("Representation sampled with shape: " + str(sampledRepresentation.shape))

        recovered = bm.recoverImage(sampledRepresentation, shape)

        print("Recovered with shape: " + str(recovered.shape))


    showResults(recovered, weights)


executeTraining()

