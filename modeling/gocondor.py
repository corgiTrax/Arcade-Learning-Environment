#!/usr/bin/env python
"""
this is an independent script that can be used to submit jobs to condor,
and run the same xxx.py file with different hyper-parameters.
the primary place you might want to change is multi_experiment().
for more detailed behaviors, please take a look at the code.
"""
def multi_experiment():
  l = [] # compose a list of arguments needed to be passed to the python script
  EXPRIMENTS=[
      ("seaquest_mixall", "Mixall")
  ]

  for (BASE_FILE_NAME, MODEL_DIR) in EXPRIMENTS:
      for dropout in ['0.2','0.3','0.4','0.5']:
          l.append(' '.join([BASE_FILE_NAME, MODEL_DIR, dropout]))

  return l

import sys, re, os, subprocess

basestr="""
# doc at : http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html
arguments = /scratch/cluster/zharucs/ale/modeling/{0}
+Group ="GRAD"
+Project ="AI_ROBOTICS"
+ProjectDescription="ale"
+GPUJob=true
Universe = vanilla

# UTCS has 18 such machine, to take a look, run 'condor_status  -constraint 'GTX1080==true' 
requirements=eldar

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

print '\n'.join(arg_str_list)
raw_input('The above arguments will be used to call %s. Confirm? Ctrl-C to quit.' % (target_py_file))

for arg_str in arg_str_list:
  submission = basestr.format(target_py_file + ' ' + arg_str)

  with open('submit.condor', 'w') as f:
    f.write(submission)

  subprocess.call(['condor_submit', 'submit.condor'])

