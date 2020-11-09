import tensorflow as tf
import numpy as np
import rnn_indices
data = rnn_indices.read_data_sets()
batch_size = 2800
unlabeled_sets = np.load('/home/asdf/Documents/juyan/igrss2018/SSRN-master/igrssdata/test_index0323.npy')
num_steps = (len(unlabeled_sets)//batch_size) + 1
test_label = np.array([])
test_label.shape = 0, 1


def rnn_label():
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
        new_saver.restore(sess, 'my-save-dir/my-model-10000')

        y = tf.get_collection('rnn_pred_label')[0]

        graph = tf.get_default_graph()

        X = graph.get_operation_by_name('X').outputs[0]

        batch_x, batch_y = data.unlabel.next_batch_test(batch_size)
        pre_test = sess.run(y, feed_dict={X: batch_x})
        pre_test = pre_test.reshape((pre_test.shape[0], 1))
        test_label = np.concatenate((test_label, pre_test), 0)
