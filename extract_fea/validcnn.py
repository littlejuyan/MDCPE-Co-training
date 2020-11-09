from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import numpy as np

import keras.backend.tensorflow_backend as KTF
import sys
sys.path.append('/home/asdf/Documents/juyan/paper/code/newcode/train/cnn')
# import loadnpzcnn
# data = loadnpzcnn.read_data_sets()
train = np.load('/home/asdf/Documents/juyan/paper/data/salinas/salinas_train.npz')


# Training Parameters
learning_rate = 0.00001
num_steps = 20000
batch_size = 100
display_step = 100

# Network Parameters
# num_input = [17, 17, 200] # MNIST data input (img shape: 28*28)
num_classes = 16 # MNIST total classes (0-9 digits)
# dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, 17, 17, 3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv3d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation

    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool3d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, keep_prob):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    # x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    x = tf.reshape(x, [-1, 17, 17, 3, 1])
    conv1 = conv3d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool3d(conv1, k=2)

    # Convolution Layer
    conv2 = conv3d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool3d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, 5*5*1*64])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)


    # Output, class prediction
    # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return fc1

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([5*5*1*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
fc1fea = conv_net(X, weights, biases, keep_prob)
# prediction = tf.nn.softmax(logits)
#
# # Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
#
#
# # Evaluate model
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()
#session = tf.Session(config=config)
with tf.Session() as sess:
    # Start training
    saver.restore(sess, "/home/asdf/Documents/juyan/paper/model/cnn0317/0317_CNN18000.ckpt")
    val_batch_x = train['cnn']
    val_batch_y = train['label']
    fc1_fea = sess.run(fc1fea, feed_dict={X: val_batch_x, Y: val_batch_y, keep_prob: 0.8})
    # for step in range(int(math.ceil(54129.0 / batch_size))):
    #     batch_x, batch_y = data.test.next_batch(batch_size)
    #      pre = sess.run(fc1fea, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
    #         # print(" Testing Accuracy= " + "{:.3f}".format(pre))
    np.save('fc1fea_18000.npy', fc1_fea)