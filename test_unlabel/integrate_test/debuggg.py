import numpy as np
import scipy.io as sio
unlabeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/cnn/unlabeled_index.npy')
labeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/cnn/labeled_index.npy')
mat_gt = sio.loadmat('/home/asdf/Documents/juyan/paper/data/salinas/cnn/Salinas_gt.mat')
GT = mat_gt['salinas_gt']
print("ok")