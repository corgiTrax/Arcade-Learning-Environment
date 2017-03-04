#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import time, sys, os
from random import randrange
from ale_python_interface import ALEInterface
import pygame, numpy as np
from IPython import embed
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

    def run(self, gc_window_drawer_func):
        black = 0, 0, 0
        last_time=time.time()
        frame_cnt=0
        # Play 10 episodes
        for episode in xrange(10):
          total_reward = 0
          while not self.ale.game_over():

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
            cur_frame_Surface = pygame.transform.scale(cur_frame_Surface, self.size)
            cur_frame_rect = cur_frame_Surface.get_rect()

            self.screen.fill(black)
            self.screen.blit(cur_frame_Surface, cur_frame_rect)
            if gc_window_drawer_func != None:
                gc_window_drawer_func(self.screen)
            pygame.display.flip()

            a = self.legal_actions[randrange(len(self.legal_actions))]
            # Apply an action and get the resulting reward
            reward = self.ale.act(a);
            total_reward += reward
          print 'Episode', episode, 'ended with score:', total_reward
          self.ale.reset_game()
