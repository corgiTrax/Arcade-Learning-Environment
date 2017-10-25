import sys, random, pygame, numpy as np
import AImodels, misc_utils as MU
from pygame.constants import RESIZABLE,DOUBLEBUF,RLEACCEL
from IPython import embed
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
import action_enums as aenum
import vip_constants as V
from aleForET import aleForET

def sample_catagorical_distribution_with_logits(logits):
    e_x = np.exp(logits - np.max(logits))
    prob = e_x / e_x.sum() # compute the softmax of logits
    picked = prob.cumsum().searchsorted(np.random.sample()) # implement weighted sampling
    return picked
def argmax_catagorical_distribution_with_logits(logits):
    return np.argmax(logits)

def draw_GHmap_callback_func(screen, pred):
    if pred['gaze'] is not None:
        GHmap = pred['gaze'].squeeze().transpose([1,0])
        GHmap = GHmap / GHmap.max() * 255.0
        GHmap = GHmap.astype(np.int8)
        GHmap = np.repeat(GHmap[..., np.newaxis], 3, axis=-1)
        s = pygame.surfarray.make_surface(GHmap)
        s.set_alpha(100)
        s = pygame.transform.scale(s, (V.SCR_W, V.SCR_H))
        screen.blit(s, (0,0))


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

    print "Received Command Line Arguments:"
    print "rom_file, model_name, model_file, mean_file = ", rom_file, model_name, model_file, mean_file
    print "resume_state_file = ", resume_state_file
    print "args_passed_to_model_initializer = ", args_passed_to_model_initializer


    # begin init

    rndseed = random.randint(0,65535)
    pygame.init()
    pygame.display.set_mode((V.SCR_W, V.SCR_H), RESIZABLE | DOUBLEBUF | RLEACCEL, 32)
    screen = pygame.display.get_surface()
    ale = aleForET(rom_file, screen, rndseed, resume_state_file)
    aimodel = getattr(AImodels,model_name)(model_file, mean_file, *args_passed_to_model_initializer)

    a = aenum.PLAYER_A_NOOP
    human_take_over = False
    print_logits = True
    draw_GHmap = True
    pred = {'gaze': None}
    ep_reward = 0

    while True:

        img_np, r, epEnd = ale.proceed_one_step(a, refresh_screen=True, fps_limit=30, 
            gc_window_drawer_func=draw_GHmap_callback_func, model_gaze_output=pred)
        # img_np, r, epEn = ale.proceed_one_step__fast__no_scr_support(a)
        pred = aimodel.predict_one(img_np)

        a = sample_catagorical_distribution_with_logits(pred['raw_logits'])

        # bookkeeping
        ep_reward += r
        if epEnd:
            print ("Episode ended. Reward: %d" % ep_reward)
            ep_reward = 0

        # --------- BEGIN keyboard event handling ------- 
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_x:
                    embed()

                if event.key == pygame.K_p: # pring logits to see if the model prediction is reasonable
                    print_logits = not print_logits
                    print "Toggle print_logits:", print_logits

                if event.key == pygame.K_ESCAPE:
                    print "K_ESCAPE pressed. Exit now."
                    sys.exit(0)

                if event.key == pygame.K_h:
                    human_take_over = not human_take_over
                    print "Toggle human play:" , human_take_over

        if print_logits and pred['raw_logits'] is not None:
            logits = pred['raw_logits'][0]
            m = np.argmax(logits)
            string = aenum.nameof(a)
            for i in range(len(logits)):
                cur = " %.1f" % logits[i]
                if i==m:
                    string += MU.color(cur,'RED')   # the action that has max logit
                elif i==a:
                    string += MU.color(cur,'GREEN') # actual action sampled
                else:
                    string += cur
            print string
            
        if human_take_over:
            key = pygame.key.get_pressed()
            a = aenum.action_map(key, ale.gamename)
        # --------- END keyboard event handling ------- 


