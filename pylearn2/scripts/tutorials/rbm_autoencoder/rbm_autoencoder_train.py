#!/usr/bin/env python
import sys
from pylearn2.config import yaml_parse

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print "usage : yaml_file_dataset yaml_file_vis_hid yaml_file_hid_hid units_layer1 units_layer2..."
        sys.exit(0)

    yaml_file_dataset = sys.argv[1]
    yaml_file_vis_hid = sys.argv[2]
    yaml_file_hid_hid = sys.argv[3]
    nb_units = map(int, sys.argv[4:])

    yaml_files = [yaml_file_vis_hid] + [yaml_file_hid_hid] * (len(nb_units) - 1)
    dataset = open(yaml_file_dataset).read()
    i = 0
    for (nvis, nhid), yaml_file in zip(zip(nb_units, nb_units[1:]), yaml_files):
        print "Training layer %d " % (i + 1,)
        data = open(yaml_file, "r").read()
        params = {"nvis": nvis, "nhid": nhid, "h_id": i,
                  "save": ("layer_%d.pkl" % (i,)), "h_id_prev": (i - 1),
                  "dataset": dataset}
        data = data % params
        train_obj = yaml_parse.load(data)
        train_obj.main_loop()
        i += 1
