#!/usr/bin/env python
import sys, random, numpy as np, os, time
import action_enums as aenum
import vip_constants as V
from aleForET import aleForET
import AImodels, misc_utils as MU
from IPython import embed

def sample_catagorical_distribution_with_logits(logits):
    e_x = np.exp(logits - np.max(logits))
    prob = e_x / e_x.sum() # compute the softmax of logits
    picked = prob.cumsum().searchsorted(np.random.sample()) # implement weighted sampling
    return picked
def argmax_catagorical_distribution_with_logits(logits):
    return np.argmax(logits)

if __name__ == "__main__":
    expected_args = [sys.argv[0], 'rom_file', 'model_name_in_AIModels.py', 'model_file', 'mean_file']
    opt_args = ['[++ resume_state_file]', '[ == args_passed_to_model_initializer]']

    if len(sys.argv) < len(expected_args):
        print 'Usage:' + ' '.join(expected_args + opt_args)
        sys.exit()

    # parse the command line args in a simple way (and prone to bugs too!)
    # TODO: use import argparse

    rom_file, model_name, model_file, mean_file  = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    _d = sys.argv.index('++') if '++' in sys.argv else -1
    resume_state_file = sys.argv[(_d+1)] if _d != -1 else None

    _d = sys.argv.index('==') if '==' in sys.argv else -1
    args_passed_to_model_initializer = sys.argv[(_d+1):] if _d != -1 else []

    MODEL_DIR = 'Expr/'+os.path.splitext(os.path.basename(rom_file))[0]
    expr = MU.ExprCreaterAndResumer(MODEL_DIR, 
        postfix="%s" % (model_name))

    print "\nReceived Command Line Arguments:"
    print "rom_file, model_name, model_file, mean_file = ", rom_file, model_name, model_file, mean_file
    print "resume_state_file = ", resume_state_file
    print "args_passed_to_model_initializer = ", args_passed_to_model_initializer
    print "\n"

    # begin init
    MU.save_GPU_mem_keras()
    rndseed = random.randint(0,65535)
    print "Random Seed: %d" % (rndseed)
    ale = aleForET(rom_file, None, rndseed, resume_state_file)
    aimodel = getattr(AImodels,model_name)(model_file, mean_file, *args_passed_to_model_initializer)

    a = aenum.PLAYER_A_NOOP
    pred = {'gaze': None}
    ep_reward = 0

    while True:
        img_np, r, epEnd = ale.proceed_one_step__fast__no_scr_support(a)
        pred = aimodel.predict_one(img_np)

        a = sample_catagorical_distribution_with_logits(pred['raw_logits'])

        diff_time = time.time()-ale._last_time
        if diff_time > 600:
            timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            print timestr, "Current Episode Score: %d" % (ale.score)
            ale._last_time=time.time()  
            if os.path.exists(expr.dir + '/STOP_ALL_EXPERIMENTS'):
                print "file STOP_ALL_EXPERIMENTS detected"
                sys.exit(0)
