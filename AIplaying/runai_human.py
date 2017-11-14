import sys, random, pygame
from pygame.constants import RESIZABLE,DOUBLEBUF,RLEACCEL
from IPython import embed
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
import action_enums as aenum
import vip_constants as V
from aleForET import aleForET

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'Usage:', sys.argv[0], 'rom_file', '[resume_state_file]'
        sys.exit()
    rom_file = sys.argv[1]
    resume_state_file = sys.argv[2] if len(sys.argv)>=3 else None
    rndseed = random.randint(0,65535)

    pygame.init()
    pygame.display.set_mode((160*V.xSCALE, 210*V.ySCALE), RESIZABLE | DOUBLEBUF | RLEACCEL, 32)
    pygame.mouse.set_visible(False)
    surf = pygame.display.get_surface()
    ale = aleForET(rom_file, surf, rndseed, resume_state_file)

    while True:
        key = pygame.key.get_pressed()
        pygame.event.pump() # need this line to get new key pressed
        a = ale.legal_actions[aenum.action_map(key, ale.gamename)]
        img_np, r, epEnd = ale.proceed_one_step(a, refresh_screen=True, fps_limit=30)

        if key[pygame.K_x]:
            embed()


