#!/usr/bin/env python
import sys, re, os, subprocess, numpy as np, string

Exp_type = sys.argv[3] # hacky, TODO

def create_bgrun_sh_content_imgOf_model(GAME_NAME):
  sh_file_content = ""
  for run in range(3):
    rom_file = "../roms/%s.bin" % GAME_NAME
    model_name = "PastKFrameOpticalFlowGaze_and_CurrentFrameAction"
    model_file = "PreMul-2ch_actionModels/" + Exp_type + "/%s.gazeIsFrom.Img+OF.hdf5" % GAME_NAME
    mean_file = "Img+OF_gazeModels/" + Exp_type + "/%s.mean.npy" % GAME_NAME
    gaze_model_file = "Img+OF_gazeModels/" + Exp_type + "/%s.hdf5" % GAME_NAME 
    optical_flow_mean_file = "Img+OF_gazeModels/" + Exp_type + "/%s.of.mean.npy" % GAME_NAME
    sh_file_content += ' '.join(['python', 'runai_noScrSupport.py', 
      rom_file, model_name, model_file, mean_file,
       '== 4 1 0', gaze_model_file, optical_flow_mean_file,
       '&\n'
       ]
      )
  sh_file_content += 'wait\n'
  return sh_file_content

def create_bgrun_sh_content_imgOnly_model(GAME_NAME):
  sh_file_content = ""
  for run in range(3):
    rom_file = "../roms/%s.bin" % GAME_NAME
    model_name = "PastKFrameGaze_and_CurrentFrameAction"
    model_file = "PreMul-2ch_actionModels/" + Exp_type + "/%s.gazeIsFrom.ImgOnly.hdf5" % GAME_NAME
    mean_file = "Img_gazeModels/" + Exp_type + "/%s.mean.npy" % GAME_NAME
    gaze_model_file = "Img_gazeModels/" + Exp_type + "/%s.hdf5" % GAME_NAME
    sh_file_content += ' '.join(['python', 'runai_noScrSupport.py',
      rom_file, model_name, model_file, mean_file,
       '== 4 1 0', gaze_model_file, 
       '&\n'
       ]
      )
  sh_file_content += 'wait\n'
  return sh_file_content

def create_bgrun_sh_content_baseline_model(GAME_NAME):
  sh_file_content = ""
  for run in range(3):
    rom_file = "../roms/%s.bin" % GAME_NAME
    model_name = "BaselineModel"
    model_file = "baseline_actionModels/" + Exp_type + "/%s.hdf5" % GAME_NAME
    mean_file = "Img+OF_gazeModels/" + Exp_type + "/%s.mean.npy" % GAME_NAME
    sh_file_content += ' '.join(['python', 'runai_noScrSupport.py',
      rom_file, model_name, model_file, mean_file,
       '&\n'
       ]
      )
  sh_file_content += 'wait\n'
  return sh_file_content

def main(bg_run_creator_func):
    basestr="""
    # doc at : http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html
    arguments = {0}
    remote_initialdir = /scratch/cluster/zharucs/ale/AIplaying/
    +Group ="GRAD"
    +Project ="AI_ROBOTICS"
    +ProjectDescription="ale"
    +GPUJob=true
    Universe = vanilla

    # UTCS has 18 such machine, to take a look, run 'condor_status  -constraint 'GTX1080==true' 
    requirements=eldar

    executable = /bin/bash 
    getenv = true
    output = CondorOutput/$(Cluster).out
    error = CondorOutput/$(Cluster).err
    log = CondorOutput/log.txt
    priority = 1
    Queue
    """
    ALL_GAME_NAMES=[
       ("seaquest")
    ]

    # ALL_GAME_NAMES=[
    #    ("breakout"),
    #    ("centipede"),
    #    ("enduro"),
    #    ("freeway"),
    #    ("mspacman"),
    #    ("riverraid"),
    #    ("seaquest"),
    #    ("venture"),
    # ]

    def fix_wrong_game_name(cmdstr):
        '''Some ale rom files do not have the same name as the model'''
        return string.replace(cmdstr, 'mspacman.bin', 'ms_pacman.bin')

    SH_FILE_DIR =  os.path.abspath('bgrun_yard')
    if not os.path.exists(SH_FILE_DIR):
      os.makedirs(SH_FILE_DIR)
    CHOSEN = [sys.argv[1]] if sys.argv[1] != 'all' else ALL_GAME_NAMES
    for GAME_NAME in CHOSEN:
        sh_file_content = bg_run_creator_func(GAME_NAME)
        sh_file_content = fix_wrong_game_name(sh_file_content)
        print sh_file_content
        raw_input('\nConfirm? Ctrl-C to quit.')

        sh_filename = "%s/%s_bgrun_%s.sh" % (SH_FILE_DIR, GAME_NAME, np.random.randint(65535))
        sh_f = open(sh_filename, 'w')
        sh_f.write(sh_file_content)

        submission = basestr.format(sh_filename)
        with open('submit.condor', 'w') as f:
          f.write(submission)

        subprocess.call(['condor_submit', 'submit.condor'])

model_to_func = {
        "imgOf": create_bgrun_sh_content_imgOf_model,
        "imgOnly":  create_bgrun_sh_content_imgOnly_model,
        "baseline": create_bgrun_sh_content_baseline_model,
        }

if len(sys.argv) < 4:
  print "Usage: %s <GAME_NAME|all> <MODEL_NAME> <EXP_TYPE>" % __file__ 
  print "'all' means run all games."
  print "Supported MODEL_NAME are: " , model_to_func.keys() 
  print "Experiment type are sub-directories in model folders"
  sys.exit(1)

if sys.argv[2] in model_to_func:
    print "Job output will be directed to folder ./CondorOutput"
    if not os.path.exists("CondorOutput"):
      os.mkdir("CondorOutput")
    main(model_to_func[sys.argv[2]])
else:
    print "ERROR: Wrong model name."
    print "Supported model names are: " , model_to_func.keys() 
    sys.exit(0)

