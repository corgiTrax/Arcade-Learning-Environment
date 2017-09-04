#! /usr/bin/env python
# this file comsumes the outputs provided by get_accuracy.py
# so it assumes a certain format of the output to work correctly

print """
summary.py reads data from stdin, so you should use pipe. Example:
ls -d <directory_pattern> | xargs -L1 get_accuracy.py | summary.py [regex_filter]

"""
import re, sys
from IPython import embed
from collections import defaultdict

class Dataset:
    def __init__(self, name):
        self.name = name
        self.model = defaultdict(lambda: defaultdict(list))

dataset_regex = re.compile('(\w+)_AAAI')
acc_regex = re.compile('^(\d+)_(.*?)_?(dr0\.\d+)_?([^ ]+)? (0\.\d+)')

filter_regex = re.compile(sys.argv[1] if len(sys.argv) == 2 else '.*')

# Collect data from stdin and build the data structure Dataset
dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for line in sys.stdin:
    datasetname_match = dataset_regex.search(line)
    line = re.sub('_run\d*','',line) # delete substring like 'run1', 'run2' to simplify processing
    acc_match = acc_regex.search(line)
    if datasetname_match:
        dataset_name=datasetname_match.group(1)
    if acc_match:
        dropout = acc_match.group(3)
        modelname = acc_match.group(2)
        if acc_match.group(4) != None:
            modelname += '_' + acc_match.group(4)
        acc = acc_match.group(5)
        dataset[dataset_name][modelname][dropout].append(float(acc))
        
# Now print the data structure Dataset. You can write code to print it in
# whatever way you want which meets your need. For example:
import numpy as np
for (name,model) in sorted(dataset.items()):
    print name
    for (modelname,paramdata) in sorted(model.items()):
        if not filter_regex.search(modelname): continue
        print modelname
        for (paramvalue, resultlist) in sorted(paramdata.items()):
            print paramvalue, np.mean(resultlist), len(resultlist)


