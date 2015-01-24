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
import re

class Fonts(DenseDesignMatrix):

    def __init__(self, kind, accept_only=None, labels_kind=None, start=None, stop=None):
        base_path = os.path.join(preprocess("${PYLEARN2_DATA_PATH}"), "fonts")

        set_file ="ds_%s.npy" % (kind,)

        full_path = os.path.join(base_path, set_file)

        data =  np.load(full_path)

        if len(data) == 2:
            data, labels = data
        else:
            labels = None


        if start is not None and stop is not None:
            data = data[start:stop]
            if labels is not None: labels = labels[start:stop]


        
        if labels is not None and accept_only is not None:
            mask = np.zeros(len(data)).astype(np.bool)
            for i, label in enumerate(labels):
                if re.match(accept_only, label):
                    mask[i] = True
                else:
                    mask[i] = False
        else:
            mask = None
    
        data = np.array(list(data))

        if labels is not None: labels = np.array(labels)
        if mask is not None:
            data = data[mask]
            if labels is not None:labels = labels[mask]
 

        if len(data.shape) == 3:
            view_c = data.shape[1], data.shape[2], 1
        elif len(data.shape) == 2:
            size = int(np.sqrt(data.shape[1]))
            view_c = size, size, 1
        view_converter = DefaultViewConverter(view_c)


        N = np.prod( data.shape[1:] )
        data = data.reshape(  (data.shape[0], N))


        if labels_kind == "letters":
            new_labels = []
            for label in labels:
                character_match = re.match('.*-([a-z])-.*', label)
                if character_match:
                    c = character_match.group(1)
                    c_id = ord(c) - ord('a')
                    new_labels.append(c_id)
                else:
                    print 'warning : unkown label...'
                    new_labels.append(0)
            labels = np.array(new_labels)
            labels = labels[:, np.newaxis].astype(np.int32)
            y_labels = 26
        else:
            y_labels = None

        print labels.shape, labels
        print data.shape
        super(Fonts, self).__init__(X=data, y=labels, view_converter=view_converter, y_labels=y_labels)

if __name__ == "__main__":
    pass
