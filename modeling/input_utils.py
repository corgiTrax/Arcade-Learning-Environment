#!/usr/bin/env python

import os, threading
import numpy as np, tensorflow as tf
from IPython import embed
from scipy import misc
import cPickle as pickle

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

