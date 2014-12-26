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

    def __init__(self, kind):
        base_path = os.path.join(preprocess("${PYLEARN2_DATA_PATH}"), "fonts")

        set_file ="ds_%s.npy" % (kind,)

        full_path = os.path.join(base_path, set_file)

        data =  np.load(full_path)
        data = np.array(list(data))
        N = np.prod( data.shape[1:] )
        data = data.reshape(  (data.shape[0], N))
        
        nb_train = data.shape[0]
        nb_train = 100 * (nb_train/100) # make it a multiple of 100
        X = data[0:nb_train]

        if len(X.shape) == 3:
            view_c = X.shape[1], X.shape[2], 1
        elif len(X.shape) == 2:
            size = int(np.sqrt(X.shape[1]))
            view_c = size, size, 1
        view_converter = DefaultViewConverter(view_c)
        super(Fonts, self).__init__(X=X, view_converter=view_converter)

if __name__ == "__main__":
    pass
