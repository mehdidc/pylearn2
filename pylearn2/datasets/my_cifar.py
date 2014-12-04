"""
Wrapper for the Adult UCI dataset:
http://archive.ics.uci.edu/ml/datasets/Adult
"""
__author__ = "Ian Goodfellow"

import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.format.target_format import convert_to_one_hot
from pylearn2.utils.string_utils import preprocess


def my_cifar(which_set):
    """
    Parameters
    ----------
    which_set : str
        'train' or 'test'

    Returns
    -------
    adult : DenseDesignMatrix
        Contains the Adult dataset.

    Notes
    -----
    This discards all examples with missing features. It would be trivial
    to modify this code to not do so, provided with a convention for how to
    treat the missing features.
    Categorical values are converted into a one-hot code.
    """

    base_path = os.path.join(preprocess("${PYLEARN2_DATA_PATH}"), "my_cifar")

    set_file = "data.npy"
    full_path = os.path.join(base_path, set_file)

    data, targets =  np.load(full_path)
    data = np.array(list(data))
    targets = np.array(list(targets))
    
    nb_train = int(data.shape[0]*0.8)
    nb_valid = int(data.shape[0]*0.1)
    nb_valid=0

    if which_set == 'train':
        X=data[0:nb_train]
        y=targets[0:nb_train]
    elif which_set == 'valid':
        X=data[nb_train:nb_train+nb_valid]
        y=targets[nb_train:nb_train+nb_valid]
    else:
        X=data[nb_train+nb_valid:]
        y=targets[nb_train+nb_valid:]
    y=np.atleast_2d(y).T
    return DenseDesignMatrix(X=X, y=y, y_labels=10)

if __name__ == "__main__":
    my_cifar(which_set='train')
    my_cifar(which_set='test')
