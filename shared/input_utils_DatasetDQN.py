#!/usr/bin/env python
import os, re, threading, time
import numpy as np
from IPython import embed
from scipy import misc
import gym
from gflag import gflag
import vip_constants as V
from copy_atari_wrappers_deprecated import wrap_dqn
import base_input_utils as BIU
import vip_constants as V

class DatasetDQN(object):
  """
  The goal of this Dataset is utilize ReplayEnv to call copy_atari_wrappers_deprecated.wrap_dqn() 
  to achieve the same preprocessing in here and in OpenAI's RL repo 'baselines'.
  
  WHAT THIS DATASET DOES: skip-4-frames, past-4-frames, take a max between past 2 frames. 
  
  (It's exactly what wrap_dqn() does. See the figure near 'Ultimately,' at https://goo.gl/qd9tmX) 
  """

  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL):
    t1=time.time()
    print("Reading all training data into memory...")
    train_imgs_raw, self.train_lbl, self.train_gaze, self.train_fid, self.train_weight = \
        BIU.read_np_parallel(LABELS_FILE_TRAIN, RESIZE_SHAPE=(84,84), preprocess_deprecated=False)
        # RESIZE_SHAPE=(84,84) because read_np_parallel() needs it to rescale gaze. 
        # It can (and should) be None after finishing read_np_parallel's TODO
    self.train_size = len(self.train_lbl)
    self.train_imgs = preprocess_imgs_via_running_ReplayEnv(train_imgs_raw)

    print("Reading all validation data into memory...")
    val_imgs_raw, self.val_lbl, self.val_gaze, self.val_fid, self.val_weight = \
        BIU.read_np_parallel(LABELS_FILE_TRAIN, RESIZE_SHAPE=(84,84), preprocess_deprecated=False)
    self.val_size = len(self.val_lbl)
    self.val_imgs = preprocess_imgs_via_running_ReplayEnv(val_imgs_raw)
    div255(self)
    print("Time spent to read train/val data: %.1fs" % (time.time()-t1))

def div255(DatasetObj): # div255 is not a member function because it's also used by other classes here
  DatasetObj.train_imgs = DatasetObj.train_imgs.astype(np.float32) / 255.0
  DatasetObj.val_imgs = DatasetObj.val_imgs.astype(np.float32) / 255.0

def preprocess_imgs_via_running_ReplayEnv(imgs_input, optional_rewards_input=None):
  env = wrap_dqn(ReplayEnv(imgs_input, optional_rewards_input), user='ReplayEnv_img')
  first_obs = env.reset()
  imgs = [np.array(first_obs)] # see below
  rewards = [0]
  while True:
    obs, reward, done, info = env.step(action=0)
    imgs.append(np.array(obs)) # convert LazyFrame (defined in copy_atari_wrappers_deprecated) to np array. TODO maybe we can keep LazyFrame and save memory. Explore.
    rewards.append(reward)
    if done:
        break
  imgs_np = np.array(imgs)
  # env.unwrapped can be used to access the innermost env, i.e., ReplayEnv
  print ("Dataset change: shape: %s -> %s dtype: %s -> %s" % (
      str(env.unwrapped.imgs.shape),str(imgs_np.shape), str(env.unwrapped.imgs.dtype),str(imgs_np.dtype)))
  return imgs_np if optional_rewards_input is None else (imgs_np, rewards)

class DatasetDQN_withMonteCarloReturn(object):
  """
  This class adds MC-return (Monte Carlo Return) data to DatasetDQN, which is computed
  from the per-frame rewards in the .asc file
  """  
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, GAZE_POS_ASC_FILE):
    t1=time.time()
    
    print("Calling read_gaze_data_asc_file()...")
    _, _, _, self.frameid2unclipped_reward, self.frameid2episode = BIU.read_gaze_data_asc_file(GAZE_POS_ASC_FILE)

    print("Reading all training data into memory...")
    train_imgs_raw, self.train_lbl, self.train_gaze, self.train_fid, self.train_weight = \
        BIU.read_np_parallel(LABELS_FILE_TRAIN, RESIZE_SHAPE=(84,84), preprocess_deprecated=False)
        # RESIZE_SHAPE=(84,84) because read_np_parallel() needs it to rescale gaze. 
        # It can (and should) be None after finishing read_np_parallel's TODO
    self.train_size = len(self.train_lbl)
    train_rewards_before_replay = self.extract_rewards_from_gaze_data(self.train_fid)
    self.train_imgs, train_rewards = preprocess_imgs_via_running_ReplayEnv(train_imgs_raw, train_rewards_before_replay)

    print("Reading all validation data into memory...")
    val_imgs_raw, self.val_lbl, self.val_gaze, self.val_fid, self.val_weight = \
        BIU.read_np_parallel(LABELS_FILE_VAL, RESIZE_SHAPE=(84,84), preprocess_deprecated=False)
    self.val_size = len(self.val_lbl)
    val_rewards_before_replay = self.extract_rewards_from_gaze_data(self.val_fid)
    self.val_imgs, val_rewards = preprocess_imgs_via_running_ReplayEnv(val_imgs_raw, val_rewards_before_replay)
    div255(self)
    print("Time spent to read train/val data: %.1fs" % (time.time()-t1))

    print("Computing Monte Carlo Return...")
    self.train_mc_return = self.compute_mc_return(self.train_fid, train_rewards, discount_factor=0.9)
    self.val_mc_return = self.compute_mc_return(self.val_fid, val_rewards, discount_factor=0.9)
    assert len(self.train_mc_return) == self.train_size and len(self.val_mc_return) == self.val_size
    self.print_baseline_loss()

  def extract_rewards_from_gaze_data(self, fids_in_label_file):
    """gaze_data in its name means self.frameid2unclipped_reward obtained from the gaze data ASC file"""
    assert self.data_is_sorted_by_time(fids_in_label_file)
    not_found = 0
    rewards = []
    for fid in fids_in_label_file:
      if fid in self.frameid2unclipped_reward:
        rewards.append(self.frameid2unclipped_reward[fid])
      else:
        not_found += 1
        rewards.append(0)
    if not_found != 0:
      print("WARNING: %d(%.1f%%) frames don't have reward data. This is not supposed to happen" % \
        not_found, float(not_found)/len(fids_in_label_file))
    return rewards

  def data_is_sorted_by_time(self, fid_list):
    # It does (1) stable sort (2) ignore and only compare the second key (i.e.frame number)
    sorted_fid_list = sorted(fid_list,cmp=lambda x,y:int(x[0]==y[0])*(x[1]-y[1])) 
    return sorted_fid_list == fid_list
    # Since later code computes Monte Carlo Return, "data_is_sorted_by_time"  assumption must be true.
    # (self.train/val_fid is a list that stores the corresponding frame ID of self.train/val_imgs)
    # If assertion failed, rewrite the dataset generation python file to satify this assumption

  def compute_mc_return(self, fids_in_label_file, rewards, discount_factor): 
    assert len(fids_in_label_file) == len(rewards)
    returns_np = np.zeros([len(fids_in_label_file)], dtype=np.float32)
    cur_return = 0.0
    total_episode_num = 0

    for i in reversed(range(len(fids_in_label_file))):
      fid = fids_in_label_file[i]
      if fid in self.frameid2episode:
        cur_return = 0
        total_episode_num += 1
      else:
        cur_return = discount_factor*cur_return + rewards[i]
      returns_np[i] = cur_return

    if total_episode_num < 10:
      print("WARNING: total number of episode recorded in this data < 10 \
        when computing Monte Carlo return. Make sure data is correct: for example, make sure\
        every end-of-life is marked as end-of-episode in the data")
    return returns_np

  def print_baseline_loss(self):
    def huber_loss(x, delta=1.0):
      return np.where(
          np.abs(x) < delta,
          np.square(x) * 0.5,
          delta * (np.abs(x) - 0.5 * delta)
      )
    print("\nbaseline loss (predicting to mean of MC return in validation): %.4f\n" % \
      np.mean(huber_loss(self.val_mc_return - np.mean(self.val_mc_return))))



class DatasetDQN_withGHmap(DatasetDQN):
  """
  This class adds GHmap (gaze heat map) data to DatasetDQN
  And the GHmap logic should be the same as gaze/input_utils.py#DatasetWithHeatmap
  See more at the comments of DatasetDQN. 
  """

  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, GAZE_POS_ASC_FILE):
    super(DatasetDQN_withGHmap, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL)
    print("Reading gaze data ASC file, and converting per-frame gaze positions to heat map...")
    self.frameid2pos, _, _, _, _ = BIU.read_gaze_data_asc_file(GAZE_POS_ASC_FILE)
    self.train_GHmap = np.zeros([self.train_size, 84, 84, 1], dtype=np.float32)
    self.val_GHmap = np.zeros([self.val_size, 84, 84, 1], dtype=np.float32)

    # Prepare train val gaze data
    print("Running BIU.convert_gaze_pos_to_heap_map() and convolution...")
    # Assign a heap map for each frame in train and val dataset
    t1 = time.time()
    bad_count, tot_count = 0, 0
    for (i,fid) in enumerate(self.train_fid):
        tot_count += len(self.frameid2pos[fid])
        bad_count += BIU.convert_gaze_pos_to_heap_map(self.frameid2pos[fid], out=self.train_GHmap[i])
    for (i,fid) in enumerate(self.val_fid):
        tot_count += len(self.frameid2pos[fid])
        bad_count += BIU.convert_gaze_pos_to_heap_map(self.frameid2pos[fid], out=self.val_GHmap[i])

    print("Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/tot_count, tot_count))
    print("'Bad' means the gaze position is outside the 160*210 screen")

    sigmaH = 28.50 * 84 / V.SCR_H
    sigmaW = 44.58 * 84 / V.SCR_W
    self.train_GHmap = BIU.preprocess_gaze_heatmap(self.train_GHmap, sigmaH, sigmaW, 0)
    self.val_GHmap = BIU.preprocess_gaze_heatmap(self.val_GHmap, sigmaH, sigmaW, 0)

    print("Normalizing the train/val heat map...")
    for i in range(len(self.train_GHmap)):
        SUM = self.train_GHmap[i].sum()
        if SUM != 0:
            self.train_GHmap[i] /= SUM

    for i in range(len(self.val_GHmap)):
        SUM = self.val_GHmap[i].sum()
        if SUM != 0:
            self.val_GHmap[i] /= SUM
    t2 = time.time()
    print("Done. BIU.convert_gaze_pos_to_heap_map() and preprocess (convolving GHmap) used: %.1fs" % (t2-t1))


class ReplayEnv(gym.Env): 
  """
  Implements a fake Atari env which replays all images in a label file.
  gym.Env doc: https://github.com/openai/gym/blob/69b677e6d8bfc0b86f586ca5ee13620b20fab90e/gym/core.py#L13
  """
  def __init__(self, imgs, rewards=None):
    """
    The argument rewards is optional. If it is given, it should be of the same length as imgs
    and contain the reward of each img.
    """
    self._init_attr()
    self.imgs = imgs
    self.rewards = rewards
    assert  (self.rewards is None) or (len(self.rewards) == len(imgs))
  def _reset(self):
    self._ptr = 0
    #  The contract of reset() is to return the initial observation.
    return self.imgs[self._ptr]
  def _step(self, action):
    assert self._ptr < self.imgs.shape[0], "Data exhausted. reset() should to be called now"
    self._ptr += 1
    obs = self.imgs[self._ptr]
    reward = self.rewards[self._ptr] if self.rewards!=None else ReplayEnv.DEFAULT_REWARD
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
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 1))
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
