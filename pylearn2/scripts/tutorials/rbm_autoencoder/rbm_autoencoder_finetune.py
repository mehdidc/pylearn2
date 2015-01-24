#!/usr/bin/env python
import sys
from pylearn2.config import yaml_parse

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "usage : yaml_file_dataset yaml_file_model layer0.pkl layer_1.pkl..."
        sys.exit(0)
    yaml_file_dataset = sys.argv[1]
    yaml_file_model = sys.argv[2]
    layer_files = sys.argv[3:]
    #layer_files_inv = list(reversed(layer_files))
    #layer_files = layer_files + layer_files_inv


    dataset = open(yaml_file_dataset, "r").read()
    layers = ["!pkl: \"%s\"" % (layer_file,) for layer_file in layer_files]
    layers_s = ",\n".join(layers)


    model = open(yaml_file_model, "r").read()
    model = model % {"layers": layers_s, "dataset": dataset}
    train_obj = yaml_parse.load(model)
    train_obj.main_loop()


