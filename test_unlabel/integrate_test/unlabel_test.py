import tensorflow as tf
import math
import predict_unlabel
import scipy.io as sio
label = predict_unlabel.predict_models()
import numpy as np
import cnn_indices
data_cnn = cnn_indices.read_data_sets()
import rnn_indices
data_rnn = rnn_indices.read_data_sets()
batch_size = 2800

updata_num_same = 200
updata_num_dcpe = 150
batch_x1 = data_cnn.unlabel.next_batch_test(batch_size)
logits_cnn, prob_cnn = label.cnn.predict1(batch_x1)
batch_x2 = data_rnn.unlabel.next_batch_test(batch_size)
logits_rnn, prob_rnn = label.rnn.predict2(batch_x2)
label_cnn = np.argmax(prob_cnn, axis=1) + 1
label_rnn = np.argmax(prob_rnn, axis=1) + 1
same_indices = np.where(label_cnn == label_rnn)
norm_rnn = np.zeros((logits_rnn.shape[0], logits_rnn.shape[1]), dtype=np.float32)
norm_cnn = np.zeros((logits_cnn.shape[0], logits_cnn.shape[1]), dtype=np.float32)
max_cnn = np.amax(logits_cnn, axis=1)
min_cnn = np.amin(logits_cnn, axis=1)
substract_cnn = [x-y for x, y in zip(max_cnn, min_cnn)]
max_rnn = np.amax(logits_rnn, axis=1)
min_rnn = np.amin(logits_rnn, axis=1)
substract_rnn = [x-y for x, y in zip(max_rnn, min_rnn)]
for i in range(logits_cnn.shape[0]):
    for j in range(logits_cnn.shape[1]):
        norm_cnn[i][j] = (logits_cnn[i][j] - min_cnn[i]) / substract_cnn[i]
        norm_rnn[i][j] = (logits_rnn[i][j] - min_rnn[i]) / substract_rnn[i]


def update_cnn():
    max_diff_cnn = [x-y for x, y in zip(norm_rnn, norm_cnn)]
    max_dcpe_cnn = np.argmax(max_diff_cnn, axis=1) + 1
    max_dcpe_cnn_list = range(2800)
    same_index = same_indices[0]
    np.random.shuffle(same_index)
    same_index = same_index[:updata_num_same]
    same_index_list = same_index.tolist()
    same_label = label_cnn[same_index_list]
    remove_same = list(set(max_dcpe_cnn_list).difference(set(same_index_list)))
    np.random.shuffle(remove_same)
    dcpe_index = remove_same[:updata_num_dcpe]
    dcpe_label = max_dcpe_cnn[dcpe_index]
    all_update_index = same_index_list + dcpe_index
    all_update_label = np.concatenate((same_label, dcpe_label), axis=0)

    unlabeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/cnn/unlabeled_index.npy')
    labeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/cnn/labeled_index.npy')
    mat_gt = sio.loadmat('/home/asdf/Documents/juyan/paper/data/salinas/cnn/Salinas_gt.mat')
    GT = mat_gt['salinas_gt']
    real_batch_index = unlabeled_sets[:2800]
    real_update_index = real_batch_index[all_update_index]
    update_labeled_sets = np.concatenate((labeled_sets, real_update_index), axis=0)
    gt_col = GT.reshape((GT.shape[0]*GT.shape[1], ))
    gt_col[real_update_index] = all_update_label
    real_gt = gt_col.reshape((GT.shape[0], GT.shape[1]))
    unlabeled_sets = unlabeled_sets[2800:]

    np.random.shuffle(update_labeled_sets)
    np.save('/home/asdf/Documents/juyan/paper/data/salinas/cnn/unlabeled_index.npy', unlabeled_sets)
    np.save('/home/asdf/Documents/juyan/paper/data/salinas/cnn/labeled_index.npy', update_labeled_sets)
    sio.savemat('/home/asdf/Documents/juyan/paper/data/salinas/cnn/Salinas_gt.mat', {'salinas_gt': real_gt})
    return ()


def update_rnn():
    max_diff_rnn = [x-y for x, y in zip(norm_cnn, norm_rnn)]
    max_dcpe_rnn = np.argmax(max_diff_rnn, axis=1) + 1
    max_dcpe_rnn_list = range(2800)
    same_index = same_indices[0]
    np.random.shuffle(same_index)
    same_index = same_index[:updata_num_same]
    same_index_list = same_index.tolist()
    same_label = label_rnn[same_index_list]

    remove_same = list(set(max_dcpe_rnn_list).difference(set(same_index_list)))
    np.random.shuffle(remove_same)
    dcpe_index = remove_same[:updata_num_dcpe]
    dcpe_label = max_dcpe_rnn[dcpe_index]
    all_update_index = same_index_list + dcpe_index
    all_update_label = np.concatenate((same_label, dcpe_label), axis=0)

    unlabeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/rnn/unlabeled_index.npy')
    labeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/rnn/labeled_index.npy')
    mat_gt = sio.loadmat('/home/asdf/Documents/juyan/paper/data/salinas/rnn/Salinas_gt.mat')
    GT = mat_gt['salinas_gt']
    real_batch_index = unlabeled_sets[:2800]
    real_update_index = real_batch_index[all_update_index]
    update_labeled_sets = np.concatenate((labeled_sets, real_update_index), axis=0)
    gt_col = GT.reshape((GT.shape[0] * GT.shape[1],))
    gt_col[real_update_index] = all_update_label
    real_gt = gt_col.reshape((GT.shape[0], GT.shape[1]))
    unlabeled_sets = unlabeled_sets[2800:]

    np.random.shuffle(update_labeled_sets)
    np.save('/home/asdf/Documents/juyan/paper/data/salinas/rnn/unlabeled_index.npy', unlabeled_sets)
    np.save('/home/asdf/Documents/juyan/paper/data/salinas/rnn/labeled_index.npy', update_labeled_sets)
    sio.savemat('/home/asdf/Documents/juyan/paper/data/salinas/rnn/Salinas_gt.mat', {'salinas_gt': real_gt})
    return ()


update_cnn()
update_rnn()




