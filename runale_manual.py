#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import time, sys
from random import randrange
from ale_python_interface import ALEInterface
import pygame 
import numpy as np
import action_enums as aenum
pygame.init()

if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

# init ale
ale = ALEInterface()
GAME_W, GAME_H = 160, 210
xSCALE, ySCALE = 6, 3

# Setting up the pygame screen Surface 
size = GAME_W * xSCALE, GAME_H * ySCALE
speed = [2, 2]
black = 0, 0, 0
screen = pygame.display.set_mode(size, pygame.RESIZABLE | pygame.DOUBLEBUF)


# Get & Set the desired settings
ale.setInt('random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = False
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', False) 
# Side Note: even if you run ale.setBool('display_screen', False) you can still have sound, i.e. set this to true
  ale.setBool('display_screen', True)

# Load the ROM file
ale.loadROM(sys.argv[1])

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()
last_time=time.time()
frame_cnt=0
# Play 10 episodes
for episode in xrange(10):
  total_reward = 0
  while not ale.game_over():

    # Display FPS
    frame_cnt+=1
    diff_time = time.time()-last_time
    if diff_time > 1.0:
#	print 'FPS: %.1f' % (frame_cnt/diff_time)
        last_time=time.time()
        frame_cnt=0
    
    time.sleep(0.02)
    
    # Show game image
    cur_frame_np = ale.getScreenRGB()
    cur_frame_Surface = pygame.surfarray.make_surface(cur_frame_np)
    cur_frame_Surface = pygame.transform.flip(cur_frame_Surface, True, False)
    cur_frame_Surface = pygame.transform.rotate(cur_frame_Surface, 90)
    cur_frame_Surface = pygame.transform.scale(cur_frame_Surface, size)
    cur_frame_rect = cur_frame_Surface.get_rect()
    screen.fill(black)
    screen.blit(cur_frame_Surface, cur_frame_rect)
    pygame.display.flip()

# random actions
#    a = legal_actions[randrange(len(legal_actions))]
    es = pygame.event.get()
    if frame_cnt == 1: 
        a_index = aenum.PLAYER_A_NOOP
    print(es)
    for e in es:
        if (e.type==pygame.KEYDOWN):
            a_index = aenum.action_map(e.key)
        elif (e.type == pygame.KEYUP):
            a_index = aenum.PLAYER_A_NOOP
            
    a = legal_actions[a_index]
    # Apply an action and get the resulting reward
    reward = ale.act(a);

    total_reward += reward
  print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()
