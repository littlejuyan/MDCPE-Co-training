import numpy as np
import os
import sys
import three_cnn
import recurrent_network
import tensorflow as tf

g_cnn = tf.Graph()
g_rnn = tf.Graph()

best_train_cnn = np.zeros((20,), dtype=np.float32)
best_train_rnn = np.zeros((20,), dtype=np.float32)
for name_num in range(20):
    print("update sample step:", name_num + 1)
    execfile('/home/asdf/Documents/juyan/paper/co_training_code/newcode/test_unlabel/integrate_test/unlabel_test.py')
    print("training cnn step:", name_num + 1)
    best_train_cnn[name_num] = three_cnn.train_cnn(name_index=name_num + 1, g_cnn=g_cnn)
    print("training rnn step:", name_num + 1)
    best_train_rnn[name_num] = recurrent_network.train_rnn(name_index=name_num + 1, g_rnn=g_rnn)
    if (name_num+1) % 5 == 0:
        np.save("/home/asdf/Documents/juyan/paper/co_training_code/newcode/model/cnn/allcnn_acc"
                + str(name_num+1) + ".npy", best_train_cnn)
        np.save("/home/asdf/Documents/juyan/paper/co_training_code/newcode/model/rnn/allrnn_acc"
                + str(name_num+1) + ".npy", best_train_rnn)