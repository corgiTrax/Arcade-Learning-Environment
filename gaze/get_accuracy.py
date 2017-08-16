#!/usr/bin/python

import os
import sys
# traverse root directory, and list directories as dirs and files as files
for d in os.listdir(sys.argv[1]):
    d = os.path.join(sys.argv[1],d)
    if os.path.isdir(d):
        if os.path.exists(d+"/log.txt"):
            f = open(d+"/log.txt",'r')
            for line in f:
                if "eval" in line:
                    score1 = line.split()[-2][0:-1]
                    score2 = line.split()[-1][0:-1]
                    print "%s %s %s" % (os.path.basename(d), score1, score2)
                    break
            f.close()
      

