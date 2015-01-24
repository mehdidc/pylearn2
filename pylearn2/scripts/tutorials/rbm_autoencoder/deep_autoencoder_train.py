#!/usr/bin/env python
import sys
from pylearn2.config import yaml_parse

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print "usage : yaml_file_dataset yaml_file_joint yaml_file_hid act_vis act_hid units_layer1 units_layer2..."
        sys.exit(0)

    yaml_file_dataset = sys.argv[1]
    yaml_file_joint = sys.argv[2]
    yaml_file_hid = sys.argv[3]
    act_vis = sys.argv[4]
    act_hid = sys.argv[5]
    nb_units = map(int, sys.argv[6:])

    joint = open(yaml_file_joint, "r").read()
    layer = open(yaml_file_hid, "r").read()

    dataset = open(yaml_file_dataset).read()
    i = 0
    layers = []
    for (nvis, nhid) in (zip(nb_units, nb_units[1:])):
        if i == 0:
            s = act_vis
        else:
            s = act_hid
        layers.append( layer % {"nvis": nvis, "nhid": nhid, "act_enc": s, "act_dec": s} )
        i += 1

    params = {"save": "model.pkl",
              "dataset": dataset,
              "layers": ",".join(layers)}
    print joint % params
    train_obj = yaml_parse.load(joint % params)
    train_obj.main_loop()
    i += 1
