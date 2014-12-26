"""
ILC dataset
"""
__author__ = "Mehdi Cherti"

import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.format.target_format import convert_to_one_hot
from pylearn2.utils.string_utils import preprocess


#view_converter = dense_design_matrix.DefaultViewConverter((28, 28, 1))

class ILC(DenseDesignMatrix):

    def __init__(self, name, binarize=True, which_set='train', train_ratio=0.8,valid_ratio=0.1):

        base_path = os.path.join(preprocess("${PYLEARN2_DATA_PATH}"), "ilc")

        set_file = "%s.npy.npz" % (name,)

        full_path = os.path.join(base_path, set_file)

        data_full =  np.load(full_path)
        if len(data_full.keys())==1:
            data = data_full[data.keys()[0]]
            y = None
        elif len(data_full.keys())==2:
            data = data_full[data_full.keys()[1]]
            y = data_full[data_full.keys()[0]]

            
            train_nb = int(data.shape[0]*train_ratio)
            valid_nb = int(data.shape[0]*valid_ratio)
            test_nb = data.shape[0]  - train_nb - valid_nb

            if which_set == 'train':
                X, y = data[0:train_nb], y[0:train_nb]
            elif which_set == 'valid':
                X, y = data[train_nb:train_nb+valid_nb], y[train_nb:train_nb+valid_nb]
            elif which_set == 'test':
                X, y = data[train_nb+valid_nb:], y[train_nb+valid_nb:]
        
        N = np.prod( X.shape[1:] )
        X = X.reshape(  (X.shape[0], N))

        view_converter = DefaultViewConverter((18, 18, 1))
        if binarize:
            X = 1.*(X > 0)
        print X.shape, y.shape
        
        print np.sum(y) 
        super(ILC, self).__init__(X=X, y=y, view_converter=view_converter)

if __name__ == "__main__":
    pass
