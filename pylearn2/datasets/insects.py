"""
.. todo::

    WRITEME
"""
import os
from theano.compat.six.moves import cPickle, xrange
import logging
_logger = logging.getLogger(__name__)

import numpy as np
import warnings
N = np
from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import contains_nan
from lasagne.datasets.insects import Insects as Insects_

class Insects(dense_design_matrix.DenseDesignMatrix):

    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : str
        One of 'train', 'test'
    center : WRITEME
    rescale : WRITEME
    gcn : float, optional
        Multiplicative constant to use for global contrast normalization.
        No global contrast normalization is applied, if None
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    toronto_prepro : WRITEME
    preprocessor : WRITEME
    """

    def __init__(self, which_set, center=False, rescale=False, gcn=None,
                 start=None, stop=None, axes=('b', 0, 1, 'c'),
                 toronto_prepro = False, preprocessor = None, grayscale=False):
        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)

        self.axes = axes

        # we define here:
        dtype = 'uint8'
        ntrain = 22000
        nvalid = 0  # artefact, we won't use it
        ntest = 2300
        

        dataset = Insects_(grayscale=grayscale)
        dataset.load()

        # we also expose the following details:
        self.img_shape = ( (3 if grayscale==False else 1), 64, 64)
        self.img_size = N.prod(self.img_shape)
        self.n_classes = 18

        # prepare loading
        x = dataset.X * 255.
        x = x.reshape( (x.shape[0], np.prod(x.shape[1:])))
        y = dataset.y
        y = dataset.y.reshape( (dataset.y.shape[0], 1) )

        # process this data
        Xs = {'train': x[0:ntrain],
              'test': x[ntrain:ntrain+ntest]}

        Ys = {'train': y[0:ntrain],
              'test': y[ntrain:ntrain+ntest]}

        X = N.cast['float32'](Xs[which_set])
        y = Ys[which_set]

        if center:
            X -= 127.5
        self.center = center

        if rescale:
            X /= 127.5
        self.rescale = rescale
        
        """
        if toronto_prepro:
            assert not center
            assert not gcn
            X = X / 255.
            if which_set == 'test':
                other = CIFAR10(which_set='train')
                oX = other.X
                oX /= 255.
                X = X - oX.mean(axis=0)
            else:
                X = X - X.mean(axis=0)
        self.toronto_prepro = toronto_prepro

        self.gcn = gcn
        if gcn is not None:
            gcn = float(gcn)
            X = global_contrast_normalize(X, scale=gcn)
        """

        if start is not None:
            # This needs to come after the prepro so that it doesn't
            # change the pixel means computed above for toronto_prepro
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop, :]
            assert X.shape[0] == y.shape[0]

#        view_converter = dense_design_matrix.DefaultViewConverter((64, 64, 3),
#                                                                  axes)

        super(Insects, self).__init__(X=X, y=y, view_converter=None,
                                      y_labels=self.n_classes)

        assert not contains_nan(self.X)

        if preprocessor:
            preprocessor.apply(self)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        # assumes no preprocessing. need to make preprocessors mark the
        # new ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'rescale'):
            self.rescale = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            for i in xrange(rval.shape[0]):
                rval[i, :] /= np.abs(rval[i, :]).max()
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval, -1., 1.)

        return rval

    def adjust_to_be_viewed_with(self, X, orig, per_example=False):
        """
        .. todo::

            WRITEME
        """
        # if the scale is set based on the data, display X oring the
        # scale determined by orig
        # assumes no preprocessing. need to make preprocessors mark
        # the new ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'rescale'):
            self.rescale = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            if per_example:
                for i in xrange(rval.shape[0]):
                    rval[i, :] /= np.abs(orig[i, :]).max()
            else:
                rval /= np.abs(orig).max()
            rval = np.clip(rval, -1., 1.)
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval, -1., 1.)

        return rval

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return Insects(which_set='test', center=self.center,
                       rescale=self.rescale, gcn=self.gcn,
                       toronto_prepro=self.toronto_prepro,
                       axes=self.axes)
