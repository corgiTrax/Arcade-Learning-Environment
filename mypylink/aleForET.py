#!/usr/bin/env python
# Author: Zhuode Liu
import time, sys, os
from random import randrange
from ale_python_interface import ALEInterface
import pygame, numpy as np
from IPython import embed
import action_enums as aenum

class aleForET:
    def __init__(self,rom_file, screen):
        self.screen = screen

        pygame.init()

        self.ale = ALEInterface()
        GAME_W, GAME_H = 160, 210
        xSCALE, ySCALE = 6, 3

        # Setting up the pygame screen Surface 
        self.size = GAME_W * xSCALE, GAME_H * ySCALE

        # Get & Set the desired settings
        self.ale.setInt('random_seed', 123)
        self.ale.setBool('sound', False) 
        self.ale.setBool('display_screen', False)

        # Load the ROM file
        self.ale.loadROM(rom_file)
        self.gamename = os.path.basename(rom_file).split('.')[0]

        # Get the list of legal actions
        self.legal_actions = self.ale.getLegalActionSet()

    def run(self, gc_window_drawer_func = None, save_screen_func = None, event_handler_func = None):
        last_time=time.time()
        frame_cnt=0
        # Play 10 episodes
        for episode in xrange(10):
          total_reward = 0
          while not self.ale.game_over():

            if event_handler_func != None:
                stop_signal, eyelink_err_code = event_handler_func()
                if stop_signal:
                    return eyelink_err_code

            # Display FPS
            frame_cnt+=1
            diff_time = time.time()-last_time
            if diff_time > 1.0:
                print 'FPS: %.1f' % (frame_cnt/diff_time)
                last_time=time.time()
                frame_cnt=0
                
            # Show game image
            cur_frame_np = self.ale.getScreenRGB()
            cur_frame_Surface = pygame.surfarray.make_surface(cur_frame_np)
            cur_frame_Surface = pygame.transform.flip(cur_frame_Surface, True, False)
            cur_frame_Surface = pygame.transform.rotate(cur_frame_Surface, 90)
            # Perform scaling directly on screen, leaving cur_frame_Surface unscaled.
            # Slightly faster than scaling cur_frame_Surface and then transfer to screen.
            pygame.transform.scale(cur_frame_Surface, self.size, self.screen)

            if gc_window_drawer_func != None:
                gc_window_drawer_func(self.screen)
            pygame.display.flip()

            # Save frame to disk (160*210, i.e. not scaled; because this is faster)
            if save_screen_func != None:
                save_screen_func(cur_frame_Surface)

            # random action
            #a = self.legal_actions[randrange(len(self.legal_actions))]
            key = pygame.key.get_pressed()
            if key[pygame.K_ESCAPE]:
                print("Exitting the game...")
            elif key[pygame.K_F1]:
                print("Pause the game...")
            elif key[pygame.K_F5]:
                print("Calibrate....")
            a_index = aenum.action_map(key, self.gamename)
            a = self.legal_actions[a_index]

            # Apply an action and get the resulting reward
            reward = self.ale.act(a);
            total_reward += reward
            # need this line to get new key pressed
            pygame.event.pump()
            # TODO: slow down game
            time.sleep(0.025)

          print 'Episode', episode, 'ended with score:', total_reward
          self.ale.reset_game()
        return 0
