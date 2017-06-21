#!/usr/bin/env python
# Author: Zhuode Liu
import time, sys, os
from ale_python_interface import ALEInterface
import pygame, numpy as np
from IPython import embed
import action_enums as aenum
import vip_constants as V

class aleForET:
    def __init__(self,rom_file, screen, rndseed, resume_state_file=None):

        # Setting up the pygame screen Surface
        pygame.init()
        self.screen = screen
        GAME_W, GAME_H = 160, 210
        self.size = GAME_W * V.xSCALE, GAME_H * V.ySCALE

        # Get & Set the desired settings
        self.ale = ALEInterface()
        self.ale.setInt('random_seed', rndseed)
        self.ale.setBool('sound', True)
        self.ale.setBool('display_screen', False)
        self.ale.setBool('color_averaging', True)
        self.ale.setFloat('repeat_action_probability', 0.0)
        
        # Load the ROM file
        self.ale.loadROM(rom_file)
        self.gamename = os.path.basename(rom_file).split('.')[0]
        
        # Get the list of legal actions
        self.legal_actions = self.ale.getLegalActionSet()
        if resume_state_file:
            self.loadALEState(resume_state_file)

    def saveALEState(self, fname):
        basedir = os.path.dirname(fname)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        pALEState = self.ale.cloneSystemState() # actually it returns an int, a memory address pointing to a C++ object ALEState
        serialized_np = self.ale.encodeState(pALEState) # this func actually takes a pointer
        np.save(fname, serialized_np)

    def loadALEState(self, fname):
        serialized_np = np.load(fname)
        pALEState = self.ale.decodeState(serialized_np) # actually it returns an int, a memory address pointing to a C++ object ALEState
        self.ale.restoreSystemState(pALEState) # this func actually takes a pointer

    def run(self, gc_window_drawer_func = None, save_screen_func = None, event_handler_func = None, record_a_and_r_func = None):
        last_time=time.time()
        frame_cnt=0
        clock = pygame.time.Clock()
        # Play 10 episodes
        for episode in xrange(10):
            total_reward = 0
            while not self.ale.game_over():
                clock.tick(30) # control FPS
                frame_cnt+=1

                key = pygame.key.get_pressed()
                if event_handler_func != None:
                    stop, eyelink_err_code, bool_drawgc = event_handler_func(key, self)
                    if stop:
                        return eyelink_err_code

                # Display FPS
                diff_time = time.time()-last_time
                if diff_time > 1.0:
                    print 'FPS: %.1f' % clock.get_fps()
                    last_time=time.time()

                # Show game image
                cur_frame_np = self.ale.getScreenRGB()
                cur_frame_Surface = pygame.surfarray.make_surface(cur_frame_np)
                cur_frame_Surface = pygame.transform.flip(cur_frame_Surface, True, False)
                cur_frame_Surface = pygame.transform.rotate(cur_frame_Surface, 90)
                # Perform scaling directly on screen, leaving cur_frame_Surface unscaled.
                # Slightly faster than scaling cur_frame_Surface and then transfer to screen.
                pygame.transform.scale(cur_frame_Surface, self.size, self.screen)

                if gc_window_drawer_func != None and bool_drawgc:
                    gc_window_drawer_func(self.screen)
                pygame.display.flip()

                # Save frame to disk (160*210, i.e. not scaled; because this is faster)
                if save_screen_func != None:
                    save_screen_func(cur_frame_Surface, frame_cnt)

                # Apply an action and get the resulting reward
                a_index = aenum.action_map(key, self.gamename)
                a = self.legal_actions[a_index]
                reward = self.ale.act(a);
                total_reward += reward
                if record_a_and_r_func != None:
                    record_a_and_r_func(a, reward)

                pygame.event.pump() # need this line to get new key pressed

            print 'Episode', episode, 'ended with score:', total_reward
            self.ale.reset_game()

        TRIAL_OK = 0 # copied from EyeLink's constant
        return TRIAL_OK

    def run_in_step_by_step_mode(self, gc_window_drawer_func = None, save_screen_func = None, event_handler_func = None, record_a_and_r_func = None):
        frame_cnt=0
        bool_drawgc = False
        clock = pygame.time.Clock()
        # Play 10 episodes
        for episode in xrange(10):
            total_reward = 0
            while not self.ale.game_over():
                # Get game image
                cur_frame_np = self.ale.getScreenRGB()
                cur_frame_Surface = pygame.surfarray.make_surface(cur_frame_np)
                cur_frame_Surface = pygame.transform.flip(cur_frame_Surface, True, False)
                cur_frame_Surface = pygame.transform.rotate(cur_frame_Surface, 90)

                frame_cnt+=1
                # Save frame to disk (160*210, i.e. not scaled; because this is faster)
                if save_screen_func != None:
                    save_screen_func(cur_frame_Surface, frame_cnt)

                key, draw_next_game_frame = None, False
                while not draw_next_game_frame:
                    clock.tick(30) # control FPS

                    key = pygame.key.get_pressed()
                    if event_handler_func != None:
                        stop, eyelink_err_code, bool_drawgc = event_handler_func(key, self)
                        if stop:
                            return eyelink_err_code
                    a_index = aenum.action_map(key, self.gamename)
                    # Not in all cases when action_map returns "NO OP" is the real action "NO OP",
                    # Only when the human press "TAB", is the real action "NO OP".
                    if (a_index == aenum.PLAYER_A_NOOP and key[pygame.K_TAB]) \
                    or  a_index != aenum.PLAYER_A_NOOP:
                        draw_next_game_frame = True

                    # Draw the image onto screen.
                    # Perform scaling directly on screen, leaving cur_frame_Surface unscaled.
                    pygame.transform.scale(cur_frame_Surface, self.size, self.screen)

                    if gc_window_drawer_func != None and bool_drawgc:
                        gc_window_drawer_func(self.screen)

                    pygame.display.flip()
                    pygame.event.pump() # need this line to get new key pressed
                
                # Apply an action and get the resulting reward
                a = self.legal_actions[a_index]
                reward = self.ale.act(a);
                total_reward += reward
                if record_a_and_r_func != None:
                    record_a_and_r_func(a, reward)

            print 'Episode', episode, 'ended with score:', total_reward
            self.ale.reset_game()
        
        TRIAL_OK = 0 # copied from EyeLink's constant
        return TRIAL_OK
