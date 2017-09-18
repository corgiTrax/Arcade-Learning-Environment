#!/usr/bin/python

import os, re
import sys

if len(sys.argv) == 1:
    print "Usage: %s directory_name [regex_filter(default='.*')]" % sys.argv[0]
    print "This is a tool to extract the final accuracy line in the output of a training process,"
    print "<directory_name> should be the 'model log output', which is MODEL_DIR in main-*.py"
    sys.exit(0)

if len(sys.argv) < 3:
    regex_filter = '.*'
else: 
    regex_filter = sys.argv[2]
regex = re.compile(regex_filter)

print sys.argv[1] + " <-- sys.argv[1]"
# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(sys.argv[1]):
    dirname = os.path.basename(root)
    if not regex.search(dirname): continue
    if "log.txt" in files:
            f = open(root +'/log.txt' ,'r')
            for line in f:
                if "eval" in line:
                    score = line.split()[-1][0:-1]
                    print "%s %s" % (dirname, score)
                    break
            f.close()
      

