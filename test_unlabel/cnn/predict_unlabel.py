import tensorflow as tf
batch_size = 2800
timesteps = 17
num_input = 12


class Predict(object):
    def __init__(self, path1, path2):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(path1)
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, path2)

    def predict1(self, batch):
        y = self.graph.get_tensor_by_name('Softmax:0')
        proba = self.graph.get_tensor_by_name('Add_1:0')
        X = self.graph.get_operation_by_name('X').outputs[0]
        keep_prob = self.graph.get_operation_by_name('keep_prob').outputs[0]
        cnn_logits, pre_pro = self.sess.run([proba, y], feed_dict={X: batch, keep_prob: 1.0})
        return cnn_logits, pre_pro

    def predict2(self, batch):
        y = self.graph.get_tensor_by_name('Softmax:0')
        proba = self.graph.get_tensor_by_name('add:0')
        X = self.graph.get_operation_by_name('X').outputs[0]
        batch = batch.reshape((batch_size, timesteps, num_input))
        pre_pro, rnn_logits = self.sess.run([y, proba], feed_dict={X: batch})
        return pre_pro, rnn_logits
        # proba = self.graph.get_tensor_by_name('add:0')
        # X = self.graph.get_operation_by_name('X').outputs[0]
        # batch = batch.reshape((batch_size, timesteps, num_input))
        # rnn_logits = self.sess.run(proba, feed_dict={X: batch})
        # return rnn_logits


def predict_models():
    class DataSets(object):
        pass
    pre_label = DataSets()
    pre_label.cnn = Predict('/home/asdf/Documents/juyan/paper/co_training_code'
                            '/newcode/model/cnn/pretrain_CNN0407.ckpt.meta',
                            '/home/asdf/Documents/juyan/paper/co_training_code/newcode/model/cnn/pretrain_CNN0407.ckpt')
    pre_label.rnn = Predict('/home/asdf/Documents/juyan/paper/co_training_code'
                            '/newcode/model/rnn/pretrain_RNN0407.ckpt.meta',
                            '/home/asdf/Documents/juyan/paper/co_training_code/newcode/model/rnn/pretrain_RNN0407.ckpt')
    return pre_label