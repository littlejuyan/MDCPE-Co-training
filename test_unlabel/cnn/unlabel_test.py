import tensorflow as tf
import math
import predict_unlabel
label = predict_unlabel.predict_models()
import numpy as np
import cnn_indices
data_cnn = cnn_indices.read_data_sets()
import rnn_indices
data_rnn = rnn_indices.read_data_sets()
batch_size = 2800

updata_num_same = 300
updata_num_dcpe = 300
batch_x1 = data_cnn.unlabel.next_batch_test(batch_size)
logits_cnn, prob_cnn = label.cnn.predict1(batch_x1)
batch_x2 = data_rnn.unlabel.next_batch_test(batch_size)
logits_rnn, prob_rnn = label.rnn.predict2(batch_x2)
label_cnn = np.argmax(prob_cnn, axis=1) + 1
label_rnn = np.argmax(prob_rnn, axis=1) + 1
same_indices = np.where(label_cnn == label_rnn)


def update_cnn():
    norm_cnn = np.zeros((logits_cnn.shape[0], logits_cnn.shape[1]), dtype=np.float32)
    max_diff_cnn = np.zeros((logits_cnn.shape[0], logits_cnn.shape[1]), dtype=np.float32)
    for i in range(logits_cnn.shape[0]):
        norm_cnn[i] = (logits_cnn[i] - np.amin(logits_cnn[i])) / float((np.amax(logits_cnn[i]) - np.amin(logits_cnn[i])))
        max_diff_cnn[i] = norm_rnn[i] - norm_cnn[i]
    max_dcpe_cnn = np.argmax(max_diff_cnn, axis=1) + 1
    max_dcpe_cnn_list = range(2800)
    np.random.shuffle(same_indices)
    same_index = same_indices[:updata_num_same]
    same_label = label_cnn[same_index]
    same_index_list = same_index.tolist()

    remove_same = list(set(max_dcpe_cnn_list).difference(set(same_index_list)))
    np.random.shuffle(remove_same)
    dcpe_index = remove_same[:updata_num_dcpe]
    dcpe_label = max_dcpe_cnn[dcpe_index]
    all_update_index = same_index + dcpe_index
    all_update_label = same_label + dcpe_label

    unlabeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/cnn/unlabeled_index.npy')
    labeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/cnn/labeled_index.npy')
    mat_gt = sio.loadmat('/home/asdf/Documents/juyan/paper/data/salinas/cnn/Salinas_gt.mat')
    GT = mat_gt['salinas_gt']
    real_batch_index = unlabeled_sets[:2800]
    real_update_index = real_batch_index[all_update_index]
    update_labeled_sets = labeled_sets + real_update_index
    gt_col = GT.ravel().tolist()
    gt_col[real_update_index] = all_update_label
    gt_col = np.array(gt_col)
    real_gt = gt_col.reshape((GT.shape[0], GT.shape[1]))
    unlabeled_sets = unlabeled_sets[2800:]

    np.save('/home/asdf/Documents/juyan/paper/data/unlabeled_index.npy', unlabeled_sets)
    np.save('/home/asdf/Documents/juyan/paper/data/labeled_index.npy', update_labeled_sets)
    sio.savemat('/home/asdf/Documents/juyan/paper/data/salinas/Salinas_gt.mat', {'salinas_gt': real_gt})
    return ()


def update_rnn():
    norm_rnn = np.zeros((logits_rnn.shape[0], logits_rnn.shape[1]), dtype=np.float32)
    max_diff_rnn = np.zeros((logits_rnn.shape[0], logits_rnn.shape[1]), dtype=np.float32)
    for i in range(logits_rnn.shape[0]):
        norm_rnn[i] = (logits_rnn[i] - np.amin(logits_rnn[i])) / float(
            (np.amax(logits_rnn[i]) - np.amin(logits_rnn[i])))
        max_diff_rnn[i] = norm_cnn[i] - norm_rnn[i]
    max_dcpe_rnn = np.argmax(max_diff_rnn, axis=1) + 1
    max_dcpe_rnn_list = range(2800)
    np.random.shuffle(same_indices)
    same_index = same_indices[:updata_num_same]
    same_label = label_rnn[same_index]
    same_index_list = same_index.tolist()

    remove_same = list(set(max_dcpe_rnn_list).difference(set(same_index_list)))
    np.random.shuffle(remove_same)
    dcpe_index = remove_same[:updata_num_dcpe]
    dcpe_label = max_dcpe_rnn[dcpe_index]
    all_update_index = same_index + dcpe_index
    all_update_label = same_label + dcpe_label

    unlabeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/rnn/unlabeled_index.npy')
    labeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/rnn/labeled_index.npy')
    mat_gt = sio.loadmat('/home/asdf/Documents/juyan/paper/data/salinas/rnn/Salinas_gt.mat')
    GT = mat_gt['salinas_gt']
    real_batch_index = unlabeled_sets[:2800]
    real_update_index = real_batch_index[all_update_index]
    update_labeled_sets = labeled_sets + real_update_index
    gt_col = GT.ravel().tolist()
    gt_col[real_update_index] = all_update_label
    gt_col = np.array(gt_col)
    real_gt = gt_col.reshape((GT.shape[0], GT.shape[1]))
    unlabeled_sets = unlabeled_sets[2800:]

    np.save('/home/asdf/Documents/juyan/paper/data/salinas/rnn/unlabeled_index.npy', unlabeled_sets)
    np.save('/home/asdf/Documents/juyan/paper/data/salinas/rnn/labeled_index.npy', update_labeled_sets)
    sio.savemat('/home/asdf/Documents/juyan/paper/data/salinas/rnn/Salinas_gt.mat', {'salinas_gt': real_gt})
    return ()




