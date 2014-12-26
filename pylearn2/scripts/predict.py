#!/usr/bin/env python

import sys

import theano
from pylearn2.utils import serial
import numpy as np

def get_prediction(model, x):
    # build theano values
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop( X )
    f = theano.function([X], Y)
    return f(x)


if __name__ == "__main__":
    #arg1 : model
    #arg2 : input file
    #arg3 : output file
    if len(sys.argv) < 4:
        print "args : model input_file output_file"
        sys.exit(0)

    # load the model
    model_path = sys.argv[1]
    model = serial.load( model_path )

    # load the input file
    x = np.load(sys.argv[2])

    # compute the predictions
    y = get_prediction(model, x)

    # save the predictions
    np.savetxt(sys.argv[3], y)
