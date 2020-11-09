from __future__ import print_function
import gzip
import os
import numpy as np
# traindata = numpy.load('/home/asdf/Documents/juyan/data/salinas/cnn/14datasets/CNN_salinas_trainsets.npz')
# testdata = numpy.load('/home/asdf/Documents/juyan/data/salinas/cnn/14datasets/CNN_salinas_testsets.npz')
# testdata = np.load('/home/asdf/Documents/juyan/data/salinas/cnn/14datasets/CNN_salinas_datasets.npz')
train = np.load('/home/asdf/Documents/juyan/paper/data/salinas/salinas_train.npz')

# label = data['cnn']

class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(train['cnn'], train['label'])
    return data_sets

