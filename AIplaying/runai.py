import sys, random, pygame
import action_enums as aenum
import vip_constants as V
from aleForET import aleForET
import AImodels
from pygame.constants import RESIZABLE,DOUBLEBUF,RLEACCEL
from IPython import embed

if __name__ == "__main__":
    expected_args = [sys.argv[0], 'rom_file', 'model_name_in_AIModels.py', 'model_file', 'mean_file']
    opt_args = ['[++ resume_state_file]', '[ == args_passed_to_model_initializer]']

    if len(sys.argv) < len(expected_args):
        print 'Usage:' + ' '.join(expected_args + opt_args)
        sys.exit()


    # parse the command line args in a simple way (and prone to bugs too!)

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
    pygame.display.set_mode((160*V.xSCALE, 210*V.ySCALE), RESIZABLE | DOUBLEBUF | RLEACCEL, 32)
    pygame.mouse.set_visible(False)
    surf = pygame.display.get_surface()
    ale = aleForET(rom_file, surf, rndseed, resume_state_file)

    aimodel = getattr(AImodels,model_name)(model_file, mean_file, *args_passed_to_model_initializer)
    a = aenum.PLAYER_A_NOOP

    while True:
        key = pygame.key.get_pressed()
        pygame.event.pump()
        img_np, r, epEnd = ale.proceed_one_step(a, refresh_screen=True, fps_limit=30)
        pred = aimodel.predict_one(img_np)
        a = pred['action']

        if key[pygame.K_x]:
            embed()
        if key[pygame.K_p]: # pring logits to see if the model prediction is reasonable
                print pred['raw_logits']


