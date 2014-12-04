"""
Wrapper for the Adult UCI dataset:
http://archive.ics.uci.edu/ml/datasets/Adult
"""
__author__ = "Ian Goodfellow"

import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.format.target_format import convert_to_one_hot
from pylearn2.utils.string_utils import preprocess


#view_converter = dense_design_matrix.DefaultViewConverter((28, 28, 1))

class Fonts(DenseDesignMatrix):

    def __init__(self, which_set, kind):
        base_path = os.path.join(preprocess("${PYLEARN2_DATA_PATH}"), "fonts")

        set_file ="ds_%s.npy" % (kind,)

        full_path = os.path.join(base_path, set_file)

        data =  np.load(full_path)
        data = np.array(list(data))
        N = np.prod( data.shape[1:] )
        data = data.reshape(  (data.shape[0], N))

        ratio_train = 1.
        ratio_valid = 0.

        nb_train = int(data.shape[0]*ratio_train)
        nb_train = 100 * (nb_train/100)

        nb_valid = int(data.shape[0]*ratio_valid)

        if which_set == 'train':
            X=data[0:nb_train]
        elif which_set == 'valid':
            X=data[nb_train:nb_train+nb_valid]
        else:
            X=data[nb_train+nb_valid:]
        print X.shape

        view_converter = DefaultViewConverter((16, 16, 1))
        super(Fonts, self).__init__(X=X, view_converter=view_converter)

if __name__ == "__main__":
    fonts(which_set='train')
    fonts(which_set='test')
