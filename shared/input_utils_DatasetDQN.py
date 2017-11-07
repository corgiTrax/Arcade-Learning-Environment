#!/usr/bin/env python
import os, re, threading, time
import numpy as np
from IPython import embed
from scipy import misc
import gym
from gflag import gflag
import vip_constants as V
from copy_atari_wrappers_deprecated import wrap_dqn
from base_input_utils import read_np_parallel

class DatasetDQN(object):
  """
  The goal of this Dataset is utilize ReplayEnv to call copy_atari_wrappers_deprecated.wrap_dqn() 
  to achieve the same preprocessing in here and in OpenAI's RL repo 'baselines'.
  
  WHAT THIS DATASET DOES: skip-4-frames, past-4-frames, take a max between past 2 frames. 
  
  (It's exactly what wrap_dqn() does. See the figure near 'Ultimately,' at https://goo.gl/qd9tmX) 
  """
  train_imgs, train_lbl, train_fid, train_size, train_weight = None, None, None, None, None
  val_imgs, val_lbl, val_fid, val_size, val_weight = None, None, None, None, None
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL):
    t1=time.time()
    print("Reading all training data into memory...")
    self.train_imgs, self.train_lbl, self.train_gaze, self.train_fid, self.train_weight = read_dataset_via_running_ReplayEnv(LABELS_FILE_TRAIN)
    self.train_size = len(self.train_lbl)
    print ("Reading all validation data into memory...")
    self.val_imgs, self.val_lbl, self.val_gaze, self.val_fid, self.val_weight = read_dataset_via_running_ReplayEnv(LABELS_FILE_VAL)
    self.val_size = len(self.val_lbl)
    self.div255()
    print ("Time spent to read train/val data: %.1fs" % (time.time()-t1))

  def div255(self):
    self.train_imgs = self.train_imgs.astype(np.float32) / 255.0
    self.val_imgs = self.val_imgs.astype(np.float32) / 255.0


def read_dataset_via_running_ReplayEnv(LABELS_FILE):
  imgs_raw, lbl, gaze, fid, weight = \
  read_np_parallel(LABELS_FILE, RESIZE_SHAPE=(84,84), preprocess_deprecated=False) 
        # RESIZE_SHAPE=(84,84) because read_np_parallel() needs it to rescale gaze. 
        # It can (and should) be None after finishing read_np_parallel's TODO
  print ("Done: read_np_parallel('%s')" % LABELS_FILE)

  env = wrap_dqn(ReplayEnv(imgs_raw), being_used_to_generate_dataset=True, scale_and_grayscale=True)
  first_obs = env.reset()
  imgs = [np.array(first_obs)] # see below
  while True:
    obs, reward, done, info = env.step(action=0)
    imgs.append(np.array(obs)) # convert LazyFram (defined in copy_atari_wrappers_deprecated) to np array. TODO maybe we can keep LazyFrame and save memory. Explore.
    if done:
        break
  imgs_np = np.array(imgs)
  # env.unwrapped can be used to access the innermost env, i.e., ReplayEnv
  print ("Dataset size change: %d -> %d " % (len(env.unwrapped.imgs),len(imgs_np)))
  return imgs_np, lbl, gaze, fid, weight





class ReplayEnv(gym.Env): 
  """
  Implements a fake Atari env which replays all images in a label file.
  gym.Env doc: https://github.com/openai/gym/blob/69b677e6d8bfc0b86f586ca5ee13620b20fab90e/gym/core.py#L13
  """
  def __init__(self, imgs):
    self._init_attr()
    self.imgs = imgs
  def _reset(self):
    self._ptr = 0
    #  The contract of reset() is to return the initial observation.
    return self.imgs[self._ptr]
  def _step(self, action):
    assert self._ptr < self.imgs.shape[0], "Data exhausted. reset() should to be called now"
    self._ptr += 1
    obs = self.imgs[self._ptr]
    reward = ReplayEnv.DEFAULT_REWARD
    done = self._ptr == self.imgs.shape[0] - 1
    info = {}
    return obs, reward, done, info
  def __getattr__(self, name):
    raise AttributeError("Attribute named '%s' is not implemented in this Env " % name +
    "because it's safer to not implement it without knowing how it's used. " + 
    "If you know how it's used you can implement it to let the program proceed.")
  def _init_attr(self):
    # Initialize some attributes that will be used by the user of this code
    # Basically, the process is like "call ReplayEnv -> see an error -> add an attribute here"
    # We only add minimal necessary attribute here. See __getattr__() for the reason.
    self._spec = ReplayEnv.make_obj(id='NoFrameskip')
    self.ale = ReplayEnv.make_obj(lives=lambda:1)
    self._action_set = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
      15, 16, 17],dtype=np.int32)
    self.action_space = gym.spaces.Discrete(len(self._action_set))
    self.get_action_meanings = lambda: [ReplayEnv.ACTION_MEANING[i] for i in self._action_set]
  @staticmethod
  def make_obj(**kwargs): # helper function used by _init_attr()
    class EmptyClass: pass
    c = EmptyClass()
    c.__dict__ = kwargs
    return c
  DEFAULT_REWARD = 1.0
  ACTION_MEANING = { # helper attribute used by _init_attr()
      0 : "NOOP",
      1 : "FIRE",
      2 : "UP",
      3 : "RIGHT",
      4 : "LEFT",
      5 : "DOWN",
      6 : "UPRIGHT",
      7 : "UPLEFT",
      8 : "DOWNRIGHT",
      9 : "DOWNLEFT",
      10 : "UPFIRE",
      11 : "RIGHTFIRE",
      12 : "LEFTFIRE",
      13 : "DOWNFIRE",
      14 : "UPRIGHTFIRE",
      15 : "UPLEFTFIRE",
      16 : "DOWNRIGHTFIRE",
      17 : "DOWNLEFTFIRE",
  }
