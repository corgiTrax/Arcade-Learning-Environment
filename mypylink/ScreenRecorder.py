#!/usr/bin/env python
# Author: Zhuode Liu
# I wrote a main function for this script, so it can be run to for testing/debuging purposes

import skimage.io
import re,sys,os,time
from aleForET import aleForET
import pygame
from IPython import embed

class ScreenRecorder:
    def __init__(self):
        rootdir = 'screen_record'
        if not os.path.exists(rootdir):
            os.mkdir(rootdir)
        indices = [int(re.match("(\d+)_", x).group(1)) for x in os.listdir(rootdir)]
        highest_indices = max(indices) if len(indices)>0 else -1
        
        # creates a dir like "5_Mar-09-12-27-59"
        self.dir = rootdir + '/' +  str(highest_indices+1) + \
            '_' + time.strftime("%b-%d-%H-%M-%S")
        os.mkdir(self.dir)
            
    def save(self, screen):
        time_val = '%.3f' % time.time()
        fname = "%s/%s.bmp" % (self.dir, str(time_val))
        pygame.image.save(screen, fname)

    def load_to_np(self, fname, size_param_used_when_playing):
        """
        This function loads an saved image into an numpy array,
        and then does the same processing step that aleForET.py does
        on the image obtained from ALE interface, so that this function 
        is trying to exactly recover what the human sees.
        *Assumption* This function assumes the processing code in aleForET.py are:
            pygame.transform.scale(cur_frame_Surface, size)
        """
        s = pygame.image.load(fname)
        s = pygame.transform.scale(s, size_param_used_when_playing)
        return pygame.surfarray.pixels3d(s).transpose(1,0,2) # transpose to (H, W, C)

    def show_img(self, img_np):
        import matplotlib.pyplot as plt
        plt.imshow(img_np)
        plt.show()

        
if __name__ == "__main__":
    import pygame, numpy as np
    pygame.init()
    GAME_W, GAME_H = 160, 210
    xSCALE, ySCALE = 6, 3
    
    if len(sys.argv)<2:
        print 'Usage: %s rom_file' % sys.argv[0]
        sys.exit(1)
    rom_file = sys.argv[1]

    # Setting up the pygame screen Surface 
    size = GAME_W * xSCALE, GAME_H * ySCALE
    screen = pygame.display.set_mode(size, pygame.RESIZABLE | pygame.DOUBLEBUF)

    ale = aleForET(rom_file, screen)
    scr_recorder = ScreenRecorder()
    ale.run(None, scr_recorder.save)
