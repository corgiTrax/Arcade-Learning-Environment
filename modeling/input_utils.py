#!/usr/bin/env python

import os, re, threading
import numpy as np, tensorflow as tf
from IPython import embed
from scipy import misc
import vip_constants as V

def frameid_from_filename(fname): 
    """ Extract '23' from '0_blahblah/23.png' """

    a, b = os.path.splitext(os.path.basename(fname))
    try:
        frameid = int(a)
    except ValueError as ex:
        raise ValueError("cannot convert filename '%s' to frame ID (an integer)" % fname)
    return frameid

def read_gaze_data_asc_file(fname):
    """ This function reads a ASC file and returns 
        a dictionary mapping frame ID to a list of gaze positions,
        a dictionary mapping frame ID to action """

    with open(fname, 'r') as f:
        lines = f.readlines()
    frameid, xpos, ypos = "BEFORE-FIRST-FRAME", None, None
    frameid2pos = {frameid: []}
    frameid2action = {frameid: None}

    for (i,line) in enumerate(lines):

        match_scr_msg = re.match("MSG\s+(\d+)\s+SCR_RECORDER FRAMEID (\d+)", line)
        if match_scr_msg: # when a new id is encountered
            timestamp, frameid = match_scr_msg.group(1), match_scr_msg.group(2)
            frameid = int(frameid)
            frameid2pos[frameid] = []
            frameid2action[frameid] = None
            continue

        freg = "[-+]?[0-9]*\.?[0-9]+" # regex for floating point numbers
        match_sample = re.match("(\d+)\s+(%s)\s+(%s)" % (freg, freg), line)
        if match_sample:
            timestamp, xpos, ypos = match_sample.group(1), match_sample.group(2), match_sample.group(3)
            xpos, ypos = float(xpos), float(ypos)
            frameid2pos[frameid].append((xpos,ypos))
            continue

        match_action = re.match("MSG\s+(\d+)\s+key_pressed atari_action (\d+)", line)
        if match_action:
            timestamp, action_label = match_action.group(1), match_action.group(2)
            if frameid2action[frameid] is None:
                frameid2action[frameid] = int(action_label)
            else:
                print "Warning: there are more than 1 action for frame id %d. Not supposed to happen." % frameid
            continue

    frameid2pos[frameid] = [] # throw out gazes after the last frame, because the game has ended but eye tracker keeps recording

    if len(frameid2pos) < 1000: # simple sanity check
        print "Warning: did you provide the correct ASC file? Because the data for only %d frames is detected" % (len(frameid2pos))
        raw_input("Press any key to continue")
    return frameid2pos, frameid2action

def convert_gaze_pos_to_heap_map(gaze_pos_list, out):
    h,w = out.shape[0], out.shape[1]
    bad_count = 0
    for (x,y) in gaze_pos_list: 
        try:
            out[int(y/V.SCR_H*h), int(x/V.SCR_W*w)] += 1
        except IndexError: # the computed X,Y position is not in range [0,160) or [0, 210)
            bad_count += 1
    out /= (np.sum(out, axis=None) + 1e-5) # normalize to a prob distribution; +1e-5 to avoid /0 when no gaze
    # using "out /=" instead of "out = out /" is a must because "out" is passed by reference (review your python knowledge)
    return bad_count

class Dataset(object):
  train_imgs, train_lbl, train_fid, train_size = None, None, None, None
  val_imgs, val_lbl, val_fid, val_size = None, None, None, None
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE):
    print "Reading all training data into memory..."
    self.train_imgs, self.train_lbl, self.train_fid = read_np_parallel(LABELS_FILE_TRAIN, RESIZE_SHAPE)
    self.train_size = len(self.train_lbl)
    print "Reading all validation data into memory..."
    self.val_imgs, self.val_lbl, self.val_fid = read_np_parallel(LABELS_FILE_VAL, RESIZE_SHAPE)
    self.val_size = len(self.val_lbl)
    print "Performing standardization (x-mean)..."
    self.standardize()
    print "Done."

  def standardize(self):
    mean = np.mean(self.train_imgs, axis=(0,1,2))
    self.train_imgs -= mean # done in-place --- "x-=mean" is faster than "x=x-mean"
    self.val_imgs -= mean


class DatasetWithGaze(Dataset):
  frameid2pos, frameid2heatmap, frameid2action_notused = None, None, None
  train_GHmap, val_GHmap = None, None # GHmap means gaze heap map
  
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, GAZE_POS_ASC_FILE, bg_prob_density):
    super(DatasetWithGaze, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)
    print "Reading gaze data ASC file, and converting per-frame gaze positions to heat map..."
    self.frameid2pos, self.frameid2action_notused = read_gaze_data_asc_file(GAZE_POS_ASC_FILE)
    self.train_GHmap = np.full([self.train_size, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1], dtype=np.float32, fill_value=bg_prob_density)
    self.val_GHmap = np.full([self.val_size, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1], dtype=np.float32, fill_value=bg_prob_density)
    self.prepare_gaze_heap_map_data()

  def prepare_gaze_heap_map_data(self):
    """Assign a heap map for each frame in train and val dataset"""
    bad_count, tot_count = 0, 0
    for (i,fid) in enumerate(self.train_fid):
        tot_count += len(self.frameid2pos[fid])
        bad_count += convert_gaze_pos_to_heap_map(self.frameid2pos[fid], out=self.train_GHmap[i])
    for (i,fid) in enumerate(self.val_fid):
        tot_count += len(self.frameid2pos[fid])
        bad_count += convert_gaze_pos_to_heap_map(self.frameid2pos[fid], out=self.val_GHmap[i])
    print "Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/tot_count, tot_count)
    print "'Bad' means the gaze position is outside the 160*210 screen"

def read_np_parallel(label_file, RESIZE_SHAPE, num_thread=6):
    """
    Read the whole dataset into memory. 
    Remember to run "imgs.nbytes" to see how much memory it uses
    Provide a label file (text file) which has "{image_path} {label}\n" per line.
    Returns a numpy array of the images, and a numpy array of labels
    """
    labels, fids = [], []
    png_files = []
    with open(label_file,'r') as f:
        for line in f:
            fname, lbl = line.strip().split(' ')
            png_files.append(fname)
            labels.append(int(lbl))
            fids.append(frameid_from_filename(fname))
    N = len(labels)
    imgs = np.empty((N,RESIZE_SHAPE[0],RESIZE_SHAPE[1],1), dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    def read_thread(PID):
        d = os.path.dirname(label_file)
        for i in range(PID, N, num_thread):
            img = misc.imread(os.path.join(d, png_files[i]), 'Y') # 'Y': grayscale
            img = misc.imresize(img, [RESIZE_SHAPE[0],RESIZE_SHAPE[1]], interp='bilinear')
            img = np.expand_dims(img, axis=2)
            img = img.astype(np.float32) / 255.0
            imgs[i,:] = img

    o=ForkJoiner(num_thread=num_thread, target=read_thread)
    o.join()
    return imgs, labels, fids

class ForkJoiner():
    def __init__(self, num_thread, target):
        self.num_thread = num_thread
        self.threads = [threading.Thread(target=target, args=[PID]) for PID in range(num_thread)]
        for t in self.threads: 
            t.start()
    def join(self):
        for t in self.threads: t.join()