import sys, random, pygame
import action_enums as aenum
import vip_constants as V
from aleForET import aleForET
from AImodels import BaselineModel
from pygame.constants import RESIZABLE,DOUBLEBUF,RLEACCEL
from IPython import embed

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print 'Usage:', sys.argv[0], 'rom_file', 'model_file', 'mean_file',  '[resume_state_file]'
        sys.exit()
    rom_file = sys.argv[1]
    model_file = sys.argv[2]
    mean_file = sys.argv[3]
    resume_state_file = sys.argv[4] if len(sys.argv)>=5 else None
    rndseed = random.randint(0,65535)

    pygame.init()
    pygame.display.set_mode((160*V.xSCALE, 210*V.ySCALE), RESIZABLE | DOUBLEBUF | RLEACCEL, 32)
    pygame.mouse.set_visible(False)
    surf = pygame.display.get_surface()
    ale = aleForET(rom_file, surf, rndseed, resume_state_file)

    aimodel = BaselineModel(model_file, mean_file)
    a = aenum.PLAYER_A_NOOP

    sd = {"pr logit":True} # states for debug proposes
    while True:
        key = pygame.key.get_pressed()
        pygame.event.pump()
        img_np, r, epEnd = ale.proceed_one_step(a, refresh_screen=True, fps_limit=30)
        pred = aimodel.predict_one(img_np)
        a = pred['action']

        if key[pygame.K_x]:
            embed()
        if key[pygame.K_p]:
            sd["pr logit"] = not sd["pr logit"]
            if sd["pr logit"]: 
                print pred['raw_logits']


