#!/usr/bin/env python
import re, sys, os
from IPython import embed
from collections import defaultdict


dataset = defaultdict(lambda: defaultdict(list))

print "Usage: %s [DIR_name] [max_data_point_count] " % sys.argv[0]
BASE_DIR = sys.argv[1]

max_data_point_count = int(sys.argv[2]) if len(sys.argv) == 3 else None
regex_episode = re.compile('Episode \d+ ended with score:')
regex_modelname = re.compile("\d+_(.*)") # match "02_BaselineModel"

print BASE_DIR + " <-- Searching for log.txt under here..."
# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(BASE_DIR):
  tokens = root.split('/') # root looks like 'Expr/breakout/02_BaselineModel'
  if regex_modelname.match(tokens[-1]):
    modelname = regex_modelname.match(tokens[-1]).group(1)
  else: 
    continue
  gamename = tokens[-2]

  if "log.txt" in files:
    f = open(root +'/log.txt' ,'r')
    for line in f:
      # line looks like "Episode 61 ended with score: 1"
        if regex_episode.search(line): 
            score = float(line.split()[-1])
            dataset[gamename][modelname].append(score)
    f.close()

# Now print the data structure.
import numpy as np
for (gamename, model) in sorted(dataset.items()):
    print "========= %s ========" % gamename
    for (modelname, resultlist) in sorted(model.items()):
      if max_data_point_count != None:
        resultlist = resultlist[:min(max_data_point_count, len(resultlist))]
      print "%4d %7.1f +- %.1f\t\t%s" % (len(resultlist), np.mean(resultlist), np.std(resultlist), modelname)

