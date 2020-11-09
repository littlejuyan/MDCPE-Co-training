""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
import os

from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
import keras.backend.tensorflow_backend as KTF
# import sys
# sys.path.append('/home/asdf/Documents/juyan/paper/code/newcode/train/rnn')
# import loadnpzrnn
# data = loadnpzrnn.read_data_sets()
train = np.load('/home/asdf/Documents/juyan/paper/data/salinas/salinas_train.npz')
# valid = np.load('/home/asdf/Documents/juyan/paper/data/salinas/salinas_valid.npz')
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
 # Execute the command (a string) in a subshell
# Training Parameters
learning_rate = 0.0001
training_steps = 50000
batch_size = 100
display_step = 200

# Network Parameters
num_input = 12 # MNIST data input (img shape: 28*28)
timesteps = 17 # timesteps
num_hidden = 100 # hidden layer num of features
num_classes = 16 # MNIST total classes (0-9 digits)

# def get_session(gpu_fraction=0.3):
#     """
#     This function is to allocate GPU memory a specific fraction
#     Assume that you have 6GB of GPU memory and want to allocate ~2GB
#     """
#
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
#
# KTF.set_session(get_session(0.5))  # using 60% of total GPU Memory
# os.system("nvidia-smi")

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.constant(0.1, shape=([num_classes, ]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, timesteps, 1)
    # Define a lstm cell with tensorflow
    gru_cell = rnn.GRUCell(num_hidden)
    # init_state = gru_cell.zero_state(dtype=tf.float32)
    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(gru_cell, x, dtype=tf.float32)
    # outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=init_state, time_major=False, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return outputs[-1]
    # return tf.matmul(states[1], weights['out']) + biases['out']


outfea = RNN(X, weights, biases)
# prediction = tf.nn.softmax(logits)
#
# # Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
# train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
# # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# # train_op = optimizer.minimize(loss_op)
#
# # Evaluate model (with test logits, for dropout to be disabled)
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# # Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:

    # Run the initializer
    saver.restore(sess, "/home/asdf/Documents/juyan/paper/model/rnn0317/0317_RNN40000.ckpt")
    batch_x = train['rnn']
    batch_y = train['label']
    batch_x = batch_x.reshape((-1, timesteps, num_input))
    pre = sess.run(outfea, feed_dict={X: batch_x, Y: batch_y})
        # print(" Testing Accuracy= " + "{:.3f}".format(pre))
    np.save('e=rnnoutfea.npy', pre)
    # batch_x = data.test.images[50000:54129]
    # batch_y = data.test.labels[50000:54129]
    #
    # batch_x = batch_x.reshape((batch_size, timesteps, num_input))
    # pre = sess.run(outfea, feed_dict={X: batch_x, Y: batch_y})
    #     # print(" Testing Accuracy= " + "{:.3f}".format(pre))
    # np.save('outfea10.npy', pre)







