#!/usr/bin/env python
# Author: Zhuode Liu
import time, sys, os
from ale_python_interface import ALEInterface
import pygame, numpy as np
from IPython import embed
import action_enums as aenum
import vip_constants as V

FRAME_RATE = 20 # important parameters, control how fast game goes 
print("****************************FrameRate:%s**********************" % FRAME_RATE) 
# now using 30 for venture and 20 for others

class aleForET:
    def __init__(self,rom_file, screen, rndseed, resume_state_file=None):
    # When you might pass None to screen:
    # You are not interested in running any functions that displays graphics
    # For example, you should only run proceed_one_step__fast__no_scr_support()
    # Otherwise, those functions uses self.screen and you will get a RuntimeError
        if screen != None:
            pygame.init()
            self.screen = screen
        GAME_W, GAME_H = 160, 210
        self.size = GAME_W * V.xSCALE, GAME_H * V.ySCALE

        # Get & Set the desired settings
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", rndseed)
        self.ale.setBool('sound', False)
        self.ale.setBool('display_screen', False)
        self.ale.setBool('color_averaging', True) #TODO!
        self.ale.setFloat('repeat_action_probability', 0.0)
        
        # Load the ROM file
        self.ale.loadROM(rom_file)
        self.gamename = os.path.basename(rom_file).split('.')[0]
        self.clock = pygame.time.Clock()
        self._last_time = time.time()
        self.score = 0
        self.episode = 0
        self.frame_cnt = 0
        
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
        np.savez(fname, state=serialized_np, score=self.score, episode=self.episode)

    def loadALEState(self, fname):
        npzfile = np.load(fname)
        serialized_np = npzfile['state']
        self.score = npzfile['score']
        self.episode = npzfile['episode']
        pALEState = self.ale.decodeState(serialized_np) # actually it returns an int, a memory address pointing to a C++ object ALEState
        self.ale.restoreSystemState(pALEState) # this func actually takes a pointer

    def proceed_one_step(self, action, refresh_screen=False, fps_limit=0, model_gaze_output=None, gc_window_drawer_func=None):
        self.clock.tick(fps_limit) # control FPS. fps_limit == 0 means no limit
        self.frame_cnt += 1

        # Display FPS
        diff_time = time.time()-self._last_time
        if diff_time > 1.0:
            print 'FPS: %.1f' % self.clock.get_fps()
            self._last_time=time.time()

        # Show game image
        cur_frame_np = self.ale.getScreenRGB()
        if refresh_screen:
            cur_frame_Surface = pygame.surfarray.make_surface(cur_frame_np)
            cur_frame_Surface = pygame.transform.flip(cur_frame_Surface, True, False)
            cur_frame_Surface = pygame.transform.rotate(cur_frame_Surface, 90)
            # Perform scaling directly on screen, leaving cur_frame_Surface unscaled.
            # Slightly faster than scaling cur_frame_Surface and then transfer to screen.
            pygame.transform.scale(cur_frame_Surface, self.size, self.screen)

            if gc_window_drawer_func != None and model_gaze_output:
                gc_window_drawer_func(self.screen, model_gaze_output)
            pygame.display.flip()

        # Apply an action and get the resulting reward
        reward = self.ale.act(action)
        self.score += reward

        return cur_frame_np, reward, self.check_episode_end_and_if_true_reset_game()

    def proceed_one_step__fast__no_scr_support(self, action):
        self.frame_cnt += 1
        cur_frame_np = self.ale.getScreenRGB()
        reward = self.ale.act(action)
        self.score += reward
        return cur_frame_np, reward, self.check_episode_end_and_if_true_reset_game()

    def check_episode_end_and_if_true_reset_game(self):
        end = self.ale.game_over()
        if end:
            print 'Episode', self.episode, 'ended with score:', self.score
            self.score = 0
            self.episode += 1
            self.ale.reset_game() 
        return end # after reset_game(),  ale.game_over()'s return value will change to false

    def run(self, gc_window_drawer_func = None, save_screen_func = None, event_handler_func = None, record_a_and_r_func = None):
        self.run_start_time = time.time() # used in alerecord_main.py
        while True:
            self.check_episode_end_and_if_true_reset_game()
            self.clock.tick(FRAME_RATE) # control FPS
            self.frame_cnt+=1

            key = pygame.key.get_pressed()
            if event_handler_func != None:
                stop, eyelink_err_code, bool_drawgc = event_handler_func(key, self)
                if stop:
                    return eyelink_err_code

            # Display FPS
            diff_time = time.time()-self._last_time
            if diff_time > 1.0:
                print 'FPS: %.1f' % self.clock.get_fps()
                self._last_time=time.time()

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
                save_screen_func(cur_frame_Surface, self.frame_cnt)

            # Apply an action and get the resulting reward
            a_index = aenum.action_map(key, self.gamename)
            a = self.legal_actions[a_index]
            reward = self.ale.act(a)
            self.score += reward
            if record_a_and_r_func != None:
                record_a_and_r_func(a, reward, self.episode, self.score)

            pygame.event.pump() # need this line to get new key pressed
        assert False, "Returning should only happen in the while True loop"

    def run_in_step_by_step_mode(self, gc_window_drawer_func = None, save_screen_func = None, event_handler_func = None, record_a_and_r_func = None):
        bool_drawgc = False
        self.run_start_time = time.time() # used in alerecord_main.py
        while True:
            self.check_episode_end_and_if_true_reset_game()
            # Get game image
            cur_frame_np = self.ale.getScreenRGB()
            cur_frame_Surface = pygame.surfarray.make_surface(cur_frame_np)
            cur_frame_Surface = pygame.transform.flip(cur_frame_Surface, True, False)
            cur_frame_Surface = pygame.transform.rotate(cur_frame_Surface, 90)

            self.frame_cnt+=1
            # Save frame to disk (160*210, i.e. not scaled; because this is faster)
            if save_screen_func != None:
                save_screen_func(cur_frame_Surface, self.frame_cnt)

            key, draw_next_game_frame = None, False
            while not draw_next_game_frame:
                self.clock.tick(FRAME_RATE) # control FPS

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
            reward = self.ale.act(a)
            self.score += reward
            if record_a_and_r_func != None:
                record_a_and_r_func(a, reward, self.episode, self.score)
        assert False, "Returning code should only be in the while True loop"
