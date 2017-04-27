#!/usr/bin/env python

import os, threading
import numpy as np, tensorflow as tf
from IPython import embed
from scipy import misc
import cPickle as pickle
import vip_constants as V

def read_gaze_data_asc_file(fname):
    """ This function reads a ASC file and returns a dictionary mapping frame ID to gaze position """

    with open(fname, 'r') as f:
        lines = f.readlines()
    frameid, xpos, ypos = "BEFORE-FIRST-FRAME", None, None
    frameid2pos = {frameid: []}

    for (i,line) in enumerate(lines):

        match_scr_msg = re.match("MSG\s+(\d+)\s+SCR_RECORDER FRAMEID (\d+)", line)
        if match_scr_msg: # when a new id is encountered
            timestamp, frameid = match_scr_msg.group(1), match_scr_msg.group(2)
            frameid = int(frameid)
            frameid2pos[frameid] = []
            continue

        freg = "[-+]?[0-9]*\.?[0-9]+" # regex for floating point numbers
        match_sample = re.match("(\d+)\s+(%s)\s+(%s)" % (freg, freg), line)
        if match_sample:
            timestamp, xpos, ypos = match_sample.group(1), match_sample.group(2), match_sample.group(3)
            xpos, ypos = float(xpos), float(ypos)
            frameid2pos[frameid].append((xpos,ypos))
            continue

    frameid2pos["AFTER-LAST-FRAME"] = frameid2pos[frameid]
    del frameid2pos[frameid] # throw out gazes after the last frame, because the game has ended but eye tracker keeps recording

    if len(frameid2pos) < 1000: # simple sanity check
        print "Warning: did you provide the correct ASC file? Because the data for only %d frames is detected" % (len(frameid2pos))
        raw_input("Press any key to continue")
    return frameid2pos

def convert_gaze_pos_to_heap_map(frameid2pos):
    """ convert every entry in dict "frameid2pos" to heap map """
    GAME_W, GAME_H = 160, 210
    frameid2heatmap = {}
    bad_count = 0
    tot_count = 0
    for (fid, gaze_pos_list) in frameid2pos.items():
        hmap = np.zeros(shape=(GAME_H, GAME_W), dtype=np.float32)
        # To convert the gaze positon from an image of size(160*xSCALE, 210*ySCALE), 
        # we can first create a heap map of size (160*xSCALE, 210*ySCALE) and rescale it to (160, 210) using interpolation in one of (‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’).
        # or we can just do something simple like here: for each gaze pos (x,y), just compute (x/xSCALE, y/ySCALE) using integer division.
        for (x,y) in gaze_pos_list: 
            try:
                # add 1/4 to this point as "heat" (we do "/4" because we want to control the input scale to CNN. And usually a point gets at most 32 gazes. so the maximum value is 32/4=8)
                hmap[int(y/V.ySCALE), int(x/V.xSCALE)] += 0.25
            except IndexError: # the computed X,Y position is not in range [0,160) and [0, 210)
                bad_count += 1
        frameid2heatmap[fid] = hmap
        tot_count += len(gaze_pos_list)
    print "Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/tot_count, tot_count)
    print "'Bad' means the gaze position is outside the 160*210 screen"
    return frameid2heatmap

class Dataset:
  train_imgs, train_lbl, train_size = None, None, None
  val_imgs, val_lbl, val_size = None, None, None
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE):
    print "Reading all training data into memory..."
    self.train_imgs, self.train_lbl = read_np_parallel(LABELS_FILE_TRAIN, SHAPE)
    self.train_size = len(self.train_lbl)
    print "Reading all validation data into memory..."
    self.val_imgs, self.val_lbl = read_np_parallel(LABELS_FILE_VAL, SHAPE)
    self.val_size = len(self.val_lbl)
    print "Performing standardization (x-mean)..."
    self.standardize()
    print "Done."

  def standardize(self):
    mean = np.mean(self.train_imgs, axis=(0,1,2))
    self.train_imgs = self.train_imgs-mean
    self.val_imgs = self.val_imgs-mean

def read_np(label_file):
    """
    Read the whole dataset into memory. 
    Remember to run "imgs.nbytes" to see how much memory it uses
    Provide a label file (text file) which has "{image_path} {label}\n" per line.
    Returns a numpy array of the images, and a numpy array of labels
    """
    labels = []
    imgs_255 = []  # misc.imread() returns an img as a 0~255 np array
    with open(label_file,'r') as f:
        d = os.path.dirname(label_file)
        for line in f:
            png_file, lbl = line.strip().split(' ')
            png = misc.imread(os.path.join(d, png_file))
            imgs_255.append(png)
            labels.append(int(lbl))
    imgs_255 = np.asarray(imgs_255, dtype=np.float32)
    imgs = imgs_255 / 255.0
    labels = np.asarray(labels, dtype=np.int32)
    return imgs, labels

def read_np_parallel(label_file, SHAPE, num_thread=10):    
    labels = []
    png_files = []
    with open(label_file,'r') as f:
        for line in f:
            fname, lbl = line.strip().split(' ')
            png_files.append(fname)
            labels.append(int(lbl))
    N = len(labels)
    imgs = np.empty([N]+list(SHAPE), dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    class ReaderThread(threading.Thread):
        def __init__(self, TID):
            super(ReaderThread, self).__init__()
            self.TID = TID
            self.daemon = True

        def run(self):
            d = os.path.dirname(label_file)
            for i in range(self.TID, N, num_thread):
                img = misc.imread(os.path.join(d, png_files[i]))
                imgs[i,:] = img.astype(np.float32) / 255.0

    threads = [ReaderThread(i) for i in range(num_thread)]
    for t in threads: t.start()
    for t in threads: t.join()
    return imgs, labels

