#!/usr/bin/env python

def multi_experiment():
    l = [] # compose a list of arguments needed to be passed to the python script
#  for dp in ["0 0.2","0 0.3","0 0.4","0 0.5"]:
#    for lr in ["0.1","0.05","0.01","0.005","0.001"]:
#        l.append(' '.join([dp,lr]))
    for dp in ["0 0.3","0 0.4","0 0.5"]: 
        for k in ["2","4","8"]:
            for s in ["1"]:
                l.append(' '.join([dp,k,s]))
    return l

import sys, re, os, subprocess

basestr="""
# doc at : http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html
arguments = /scratch/cluster/zharucs/ale/gaze/{0}
remote_initialdir = /scratch/cluster/zharucs/ale/gaze/
+Group ="GRAD"
+Project ="AI_ROBOTICS"
+ProjectDescription="ale"
+GPUJob=true
Universe = vanilla

# UTCS has 18 such machine, to take a look, run 'condor_status  -constraint 'GTX1080==true' 
Requirements=(TARGET.GTX1080== true)

executable = /u/zharucs/anaconda2/bin/ipython 
getenv = true
output = CondorOutput/$(Cluster).out
error = CondorOutput/$(Cluster).err
log = CondorOutput/log.txt
Queue
"""

if len(sys.argv) < 2:
  print "Usage: %s target_py_file" % __file__ 
  sys.exit(1)
if not os.path.exists("CondorOutput"):
  os.mkdir("CondorOutput")
print "Job output will be directed to folder ./CondorOutput"

target_py_file = sys.argv[1]

arg_str_list = multi_experiment()

for arg_str in arg_str_list:
  submission = basestr.format(target_py_file + ' ' + arg_str)

  with open('submit.condor', 'w') as f:
    f.write(submission)

  subprocess.call(['condor_submit', 'submit.condor'])

