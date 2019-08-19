#!/usr/bin/env python
"""
this is an independent script that can be used to submit jobs to condor,
and run the same xxx.py file with different hyper-parameters.
the primary place you might want to change is multi_experiment().
for more detailed behaviors, please take a look at the code.
"""
def multi_experiment():
    l = [] # compose a list of arguments needed to be passed to the python script

    for game in [
#                  "alien",
#                 "asterix",
#                 "bank_heist",
#                 "berzerk",
#                 "breakout",
#                 "centipede",
#                 "demon_attack",
#                 "enduro",
#                 "freeway",
#                 "frostbite",
#                 "hero",
#                 "montezuma_revenge",
#                 "ms_pacman",
#                 "name_this_game",
#                 "phoenix",
#                 "riverraid",
#                 "road_runner",
#                 "seaquest",
#                 "space_invaders",
                 "venture"
                ]:
#
#        for dropout in ['0.1','0.2','0.3','0.4','0.5']:
        for dropout in ['0.2']:
            l.append(' '.join([game, '0', dropout]))
#    for param in [
#                   "ms_pacman 0 0.4",
#                   "breakout 0 0.1",
#                   "bank_hesit 0 0.2",
#                  ]:
#        l.append(' '.join([param]))


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
requirements=eldar

executable = /u/zharucs/anaconda2/bin/ipython 
getenv = true
output = CondorOutput/$(Cluster).out
error = CondorOutput/$(Cluster).err
log = CondorOutput/log.txt
Queue
"""

#Requirements=(TARGET.GTX1080== true)
#requirements=eldar
#+GPUJob=true

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

