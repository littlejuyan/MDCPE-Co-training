# -*- coding: utf-8 -*-
import numpy as np

import scipy.io as sio


def sampling(groundTruth):              #divide dataset into train and test datasets
    labeled = {}
    test = {}
    valid = {}
    m = max(groundTruth)
    labeled_indices = []
    test_indices = []
    valid_indices = []
    unlabeled_indices = []
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i]
        if i == 0:
            np.random.shuffle(indices)
            unlabeled_indices = indices
        else:
            np.random.shuffle(indices)
            nb_val = int(0.8 * len(indices))
            test[i] = indices[:nb_val]
            valid[i] = indices[nb_val:int(0.87*len(indices))]
            labeled[i] = indices[int(0.87*len(indices)):]
            labeled_indices += labeled[i]
            test_indices += test[i]
            valid_indices += valid[i]
    np.random.shuffle(labeled_indices)
    np.random.shuffle(test_indices)
    np.random.shuffle(valid_indices)

    return labeled_indices, test_indices, valid_indices, unlabeled_indices


mat_gt = sio.loadmat("/home/asdf/Documents/juyan/paper/data/salinas/Salinas_gt.mat")
gt_IN = mat_gt['salinas_gt']
new_gt_IN = gt_IN

gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)


labeled_indices, test_indices, valid_indices, unlabeled_indices = sampling(gt)
print(len(labeled_indices))
print(len(valid_indices))
print(len(unlabeled_indices))
print(len(test_indices))
np.save('/home/asdf/Documents/juyan/paper/data/labeled_index.npy', labeled_indices)
np.save('/home/asdf/Documents/juyan/paper/data/valid_index.npy', valid_indices)
np.save('/home/asdf/Documents/juyan/paper/data/unlabeled_index.npy', unlabeled_indices)
np.save('/home/asdf/Documents/juyan/paper/data/test_index.npy', test_indices)





