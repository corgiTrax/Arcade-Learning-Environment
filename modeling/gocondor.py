#!/usr/bin/env python
def multi_experiment():
  l = [] # compose a list of arguments needed to be passed to the python script
  EXPRIMENTS=[
      ("breakout_{92}val","breakout_AAAI"),
      ("centipede_{78_80}val","centipede_AAAI"),
      ("enduro_{98_103}val","enduro_AAAI"),
      ("freeway_{72}val","freeway_AAAI"),
      ("mspacman_{71_76}val","mspacman_AAAI"),
      ("riverraid_{95_99}val","riverraid_AAAI"),
      ("seaquest_{70_75}val","seaquest_AAAI"),
      ("venture_{100_101}val","venture_AAAI")
  ]

  for (BASE_FILE_NAME, MODEL_DIR) in EXPRIMENTS:
      for dropout in ['0','0.1', '0.2', '0.3', '0.4', '0.5']:
          ABS_BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/" + BASE_FILE_NAME
          l.append(' '.join([ABS_BASE_FILE_NAME, MODEL_DIR, dropout]))

  return l

import sys, re, os, subprocess

basestr="""
# doc at : http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html
arguments = /scratch/cluster/zhuode93/ale/modeling/{0}
+Group ="GRAD"
+Project ="AI_ROBOTICS"
+ProjectDescription="ale"
+GPUJob=true
Universe = vanilla

# UTCS has 18 such machine, to take a look, run 'condor_status  -constraint 'GTX1080==true' 
Requirements=(TARGET.GTX1080== true)

executable = /u/zhuode93/anaconda2/bin/ipython 
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

print '\n'.join(arg_str_list)
raw_input('Confirm? Ctrl-C to quit.')

for arg_str in arg_str_list:
  submission = basestr.format(target_py_file + ' ' + arg_str)

  with open('submit.condor', 'w') as f:
    f.write(submission)

  subprocess.call(['condor_submit', 'submit.condor'])

