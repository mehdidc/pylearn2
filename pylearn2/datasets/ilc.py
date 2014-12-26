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

class ILC(DenseDesignMatrix):

    def __init__(self):
        base_path = os.path.join(preprocess("${PYLEARN2_DATA_PATH}"), "ilc")

        set_file ="ilc.npy"

        full_path = os.path.join(base_path, set_file)

        data =  np.load(full_path)
        data = np.array(list(data))
        N = np.prod( data.shape[1:] )
        data = data.reshape(  (data.shape[0], N))

        view_converter = DefaultViewConverter((18, 18, 1)) # will not work...
        X = data
        X = 1.*(X > 0)
        super(ILC, self).__init__(X=X, view_converter=view_converter)

if __name__ == "__main__":
    pass
