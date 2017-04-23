#!/usr/bin/env python

import os
import numpy as np, tensorflow as tf
from IPython import embed
from scipy import misc

class Dataset:
  train_imgs, train_lbl, train_size = None, None, None
  val_imgs, val_lbl, val_size = None, None, None
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL):
    print "Reading all training data into memory..."
    self.train_imgs, self.train_lbl = read_np(LABELS_FILE_TRAIN)
    self.train_size = len(self.train_lbl)
    print "Reading all validation data into memory..."
    self.val_imgs, self.val_lbl = read_np(LABELS_FILE_VAL)
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
        for line in f:
            png_file, lbl = line.strip().split(' ')
            png = misc.imread(os.path.join(os.path.dirname(label_file), png_file))
            imgs_255.append(png)
            labels.append(int(lbl))
    imgs_255 = np.asarray(imgs_255, dtype=np.float32)
    imgs = imgs_255 / 255.0
    labels = np.asarray(labels, dtype=np.int32)
    return imgs, labels

