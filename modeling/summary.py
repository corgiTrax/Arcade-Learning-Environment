#! /usr/bin/env python
# this file comsumes the outputs provided by get_accuracy.py
# so it assumes a certain format of the output to work correctly
import re, sys
from IPython import embed
from collections import defaultdict

class Dataset:
    def __init__(self, name):
        self.name = name
        self.model = defaultdict(lambda: defaultdict(list))

dataset_regex = re.compile('(\w+)_AAAI')
acc_regex = re.compile('^(\d+)_(\w+)_(dr0\.\d+)_?(\w+)? (0\.\d+)')

# If no argument is provided, read from stdin. This is useful in piping commands
f_ = sys.stdin if len(sys.argv) == 1 else open(sys.argv[1], 'r')

dataset = []
for line in f_:
    datasetname_match = dataset_regex.search(line)
    line = re.sub('_run\d*','',line) # delete substring like 'run1', 'run2' to simplify processing
    acc_match = acc_regex.search(line)
    if datasetname_match:
        dataset.append(Dataset(datasetname_match.group(1)))
    if acc_match:
        dropout = acc_match.group(3)
        modelname = acc_match.group(2)
        if acc_match.group(4) != None:
            modelname += '_' + acc_match.group(4)
        acc = acc_match.group(5)
        dataset[-1].model[modelname][dropout].append(float(acc))
        

import numpy as np
for d in dataset:
    print d.name
    for (modelname,paramdata) in sorted(d.model.items()):
        print modelname
        for (paramvalue, resultlist) in sorted(paramdata.items()):
            print paramvalue, np.mean(resultlist), len(resultlist)


