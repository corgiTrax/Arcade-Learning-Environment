#!/usr/bin/env python

import os, re, threading, time, tarfile
import numpy as np
import ipdb
from IPython import embed
from scipy import misc
from ast import literal_eval
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import sys
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
import base_input_utils as BIU
import vip_constants as V

class DatasetWithHeatmap(BIU.Dataset):
  frameid2pos, frameid2action_notused = None, None
  train_GHmap, val_GHmap = None, None # GHmap means gaze heap map
  
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, HEATMAP_SHAPE, GAZE_POS_ASC_FILE):
    super(DatasetWithHeatmap, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)
    print "Reading gaze data ASC file, and converting per-frame gaze positions to heat map..."
    self.frameid2pos, self.frameid2action_notused = BIU.read_gaze_data_asc_file(GAZE_POS_ASC_FILE)
    self.train_GHmap = np.zeros([self.train_size, HEATMAP_SHAPE, HEATMAP_SHAPE, 1], dtype=np.float16)
    self.val_GHmap = np.zeros([self.val_size, HEATMAP_SHAPE, HEATMAP_SHAPE, 1], dtype=np.float16)

    # Prepare train val gaze data
    print "Running BIU.convert_gaze_pos_to_heap_map() and convolution..."
    # Assign a heap map for each frame in train and val dataset
    t1 = time.time()
    bad_count, tot_count = 0, 0
    for (i,fid) in enumerate(self.train_fid):
        tot_count += len(self.frameid2pos[fid])
        bad_count += BIU.convert_gaze_pos_to_heap_map(self.frameid2pos[fid], out=self.train_GHmap[i])
    for (i,fid) in enumerate(self.val_fid):
        tot_count += len(self.frameid2pos[fid])
        bad_count += BIU.convert_gaze_pos_to_heap_map(self.frameid2pos[fid], out=self.val_GHmap[i])

    print "Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/tot_count, tot_count)    
    print "'Bad' means the gaze position is outside the 160*210 screen"

    sigmaH = 28.50 * HEATMAP_SHAPE / V.SCR_H
    sigmaW = 44.58 * HEATMAP_SHAPE / V.SCR_W
    self.train_GHmap = BIU.preprocess_gaze_heatmap(self.train_GHmap, sigmaH, sigmaW, 0)
    self.val_GHmap = BIU.preprocess_gaze_heatmap(self.val_GHmap, sigmaH, sigmaW, 0)

    print "Normalizing the train/val heat map..."
    for i in range(len(self.train_GHmap)):
        SUM = self.train_GHmap[i].sum()
        if SUM != 0:
            self.train_GHmap[i] /= SUM

    for i in range(len(self.val_GHmap)):
        SUM = self.val_GHmap[i].sum()
        if SUM != 0:
            self.val_GHmap[i] /= SUM
    print "Done. BIU.convert_gaze_pos_to_heap_map() and convolution used: %.1fs" % (time.time()-t1)

class DatasetWithHeatmap_PastKFrames(DatasetWithHeatmap):
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, HEATMAP_SHAPE, GAZE_POS_ASC_FILE, K, stride=1, before=0):
    super(DatasetWithHeatmap_PastKFrames, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, HEATMAP_SHAPE, GAZE_POS_ASC_FILE)
    # delete the following line when merge with lzd's input_util.py in the future, to save memory
    # self.train_imgs_bak, self.val_imgs_bak = self.train_imgs, self.val_imgs

    t1=time.time()
    self.train_imgs = BIU.transform_to_past_K_frames(self.train_imgs, K, stride, before)
    # If not using 3DConv, comment the line below
    #self.train_imgs = np.expand_dims(self.train_imgs, axis=4)
    self.val_imgs = BIU.transform_to_past_K_frames(self.val_imgs, K, stride, before)
    #self.val_imgs = np.expand_dims(self.val_imgs, axis=4)

    # Trim labels. This is assuming the labels align with the training examples from the back!!
    # Could cause the model unable to train  if this assumption does not hold
    self.train_lbl = self.train_lbl[-self.train_imgs.shape[0]:]
    self.val_lbl = self.val_lbl[-self.val_imgs.shape[0]:]

    self.train_size = len(self.train_lbl)
    self.val_size = len(self.val_lbl)

    self.train_gaze = self.train_gaze[-self.train_imgs.shape[0]:]
    self.val_gaze = self.val_gaze[-self.val_imgs.shape[0]:]

    self.train_fid = self.train_fid[-self.train_imgs.shape[0]:]
    self.val_fid = self.val_fid[-self.val_imgs.shape[0]:]

    self.train_GHmap = self.train_GHmap[-self.train_imgs.shape[0]:]
    self.val_GHmap = self.val_GHmap[-self.val_imgs.shape[0]:]

    self.train_weight = self.train_weight[-self.train_imgs.shape[0]:]
    self.val_weight = self.val_weight[-self.val_imgs.shape[0]:]

    print "Time spent to transform train/val data to past K frames: %.1fs" % (time.time()-t1)
    
    
class Dataset_OpticalFlow(object):
    train_flow, val_flow = None, None

    def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE):
        t1 = time.time()
        print "Reading train optical flow images into memory..."
        self.train_flow = read_optical_flow(LABELS_FILE_TRAIN, RESIZE_SHAPE)
        print "Reading val optical flow images into memory..."
        self.val_flow = read_optical_flow(LABELS_FILE_VAL, RESIZE_SHAPE)
        print "Time spent to read train/val optical flow data: %.1fs" % (time.time()-t1)
        
        print "Performing standardization (x-mean)..."
        self.standardize_opticalflow()
        print "Done."

    def standardize_opticalflow(self):
        self.mean = np.mean(self.train_flow, axis=(0,1,2))
        self.train_flow -= self.mean # done in-place --- "x-=mean" is faster than "x=x-mean"
        self.val_flow -= self.mean


class Dataset_OpticalFlow_PastKFrames(Dataset_OpticalFlow):
    def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, K, stride=1, before=0):
        super(Dataset_OpticalFlow_PastKFrames, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)

        # With past k optical flow images
        t1 = time.time()
        print "Transforming optical flow images to past k frames..."
        self.train_flow = BIU.transform_to_past_K_frames(self.train_flow, K, stride, before)
        self.val_flow = BIU.transform_to_past_K_frames(self.val_flow, K, stride, before)
        print "Time spent to transform tran/val optical flow images to past k frames: %.1fs" % (time.time()-t1)
    
class Dataset_BottomUp(object):
    train_bottom, val_bottom = None, None

    def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE):
        t1 = time.time()
        print "Reading train bottom up images into memory..."
        self.train_bottom = read_bottom_up(LABELS_FILE_TRAIN, RESIZE_SHAPE)
        print "Reading val bottom up images into memory..."
        self.val_bottom = read_bottom_up(LABELS_FILE_VAL, RESIZE_SHAPE)
        print "Time spent to read train/val bottom up data: %.1fs" % (time.time()-t1)
        
        print "Performing standardization (x-mean)..."
        self.standardize_bottomup()
        print "Done."

    def standardize_bottomup(self):
        mean = np.mean(self.train_bottom, axis=(0,1,2))
        self.train_bottom -= mean # done in-place --- "x-=mean" is faster than "x=x-mean"
        self.val_bottom -= mean

class Dataset_BottomUp_PastKFrames(Dataset_BottomUp):
    def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, K, stride=1, before=0):
        super(Dataset_BottomUp_PastKFrames, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)
    
        t1 = time.time()
        print "Transforming bottom up images to past k frames..."
        self.train_bottom = BIU.transform_to_past_K_frames(self.train_bottom, K, stride, before)
        self.val_bottom = BIU.transform_to_past_K_frames(self.val_bottom, K, stride, before)
        print "Time spent to transform tran/val bottom up images to past k frames: %.1fs" % (time.time()-t1)

def read_optical_flow(label_file, RESIZE_SHAPE, num_thread=6):
    png_files = []
    with open(label_file,'r') as f:
        for line in f:
            line=line.strip()
            if line.startswith("#") or line == "": 
                continue # skip comments or empty lines
            fname, lbl, x, y, _ = line.split(' ')
            png_files.append(fname)

    N = len(png_files)
    imgs = np.empty((N,RESIZE_SHAPE[0],RESIZE_SHAPE[1],1), dtype=np.float16)

    def read_thread(PID):
        d = os.path.dirname(label_file)
        for i in range(PID, N, num_thread):
            try:
                img = misc.imread(os.path.join(d+'/optical_flow', png_files[i]))
            except IOError:
                img = np.zeros((RESIZE_SHAPE[0],RESIZE_SHAPE[1]), dtype=np.float16)
                print "Warning: %s has no optical flow image. Set to zero." % png_files[i]
            img = np.expand_dims(img, axis=2)
            img = img.astype(np.float16) / 255.0 # normalize image to [0,1]            
            imgs[i,:] = img

    o=BIU.ForkJoiner(num_thread=num_thread, target=read_thread)
    o.join()

    return imgs

def read_bottom_up(label_file, RESIZE_SHAPE, num_thread=6):
    png_files = []
    with open(label_file,'r') as f:
        for line in f:
            line=line.strip()
            if line.startswith("#") or line == "": 
                continue # skip comments or empty lines
            fname, lbl, x, y, _ = line.split(' ')
            png_files.append(fname)

    N = len(png_files)
    imgs = np.empty((N,RESIZE_SHAPE[0],RESIZE_SHAPE[1],1), dtype=np.float16)

    def read_thread(PID):
        d = os.path.dirname(label_file)
        for i in range(PID, N, num_thread):
            try:
                img = misc.imread(os.path.join(d+'/bottom_up', png_files[i]))
            except IOError:
                img = np.zeros((RESIZE_SHAPE[0],RESIZE_SHAPE[1]), dtype=np.float16)
                print "Warning: %s has no bottom up image. Set to zero." % png_files[i]
            img = np.expand_dims(img, axis=2)
            img = img.astype(np.float16) / 255.0 # normalize image to [0,1]            
            imgs[i,:] = img

    o=BIU.ForkJoiner(num_thread=num_thread, target=read_thread)
    o.join()

    return imgs

def read_result_data(result_file, RESIZE_SHAPE):
    """
    Read the predicted gaze positions from a txt file
    return a dictionary that maps fid to predicted gaze position
    """
    fid = "BEFORE-FIRST-FRAME"
    predicts = {fid: (-1,-1)}
    with open(result_file,'r') as f:
        for line in f:
            line=line.strip()
            if line.startswith("#") or line == "": 
                continue # skip comments or empty lines
            fid, x, y = line.split(' ')
            fid = literal_eval(fid)
            predicts[fid] = (float(x)*V.SCR_W/RESIZE_SHAPE[1], float(y)*V.SCR_H/RESIZE_SHAPE[0])
    return predicts

def save_heatmap_png_files(frameids, heatmaps, dataset, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fid = "BEFORE-FIRST-FRAME"
    frameid2heatmap = {fid: []}
    for i in range(len(frameids)):
        # heatmaps[i] = heatmaps[i]/heatmaps[i].max() * 255.0
        frameid2heatmap[(frameids[i][0],frameids[i][1])] = heatmaps[i,:,:,0]
    
    hashID2name = {0: []}
    for i in range(len(dataset)):
        _, person, number, date = dataset[i].split('_')
        UTID = person + '_' + number
        save_path = save_dir + dataset[i]
        hashID2name[hash(UTID)] = [save_path, UTID+'_']
        if not os.path.exists(save_path):
            os.mkdir(save_path) 

    m = cm.ScalarMappable(cmap='jet')

#    # no multithread version
#    print "Convolving heatmaps and saving into png files..."
#    t1 = time.time()
#    for fid in frameid2heatmap:
#        if fid == 'BEFORE-FIRST-FRAME':
#            continue
#
#        pic = convolve(frameid2heatmap[fid], Gaussian2DKernel(stddev=1))
#        pic = m.to_rgba(pic)[:,:,:3]
#        plt.imsave(hashID2name[fid[0]][0]+'/' + hashID2name[fid[0]][1] + str(fid[1]) + '.png', pic)
#    print "Done. Time spent to save heatmaps: %.1fs" % (time.time()-t1)

    # multithread version, this is not done yet
    print "Convolving heatmaps and saving into png files..."
    t1 = time.time()
    num_thread = 6
    def read_thread(PID):
        for fid in frameid2heatmap:
            if fid == 'BEFORE-FIRST-FRAME':
                continue
            if int(fid[1]) % num_thread == PID:
                pic = convolve(frameid2heatmap[fid], Gaussian2DKernel(stddev=1))
                pic = m.to_rgba(pic)[:,:,:3]
                plt.imsave(hashID2name[fid[0]][0]+'/' + hashID2name[fid[0]][1] + str(fid[1]) + '.png', pic)

    o=BIU.ForkJoiner(num_thread=num_thread, target=read_thread)
    o.join()
    print "Done. Time spent to save heatmaps: %.1fs" % (time.time()-t1)

    print "Tar the png files..."
    t2 = time.time()
    for hashID in hashID2name:
        if hashID2name[hashID]:
            make_targz_one_by_one(hashID2name[hashID][0] + '.tar.bz2', hashID2name[hashID][0])
    print "Done. Time spent to tar files: %.1fs" % (time.time()-t2)

def make_targz_one_by_one(output_filename, source_dir): 
    tar = tarfile.open(output_filename,"w:gz")
    for root,dir,files in os.walk(source_dir):
        for file in files:
            pathfile = os.path.join(root, file)
            tar.add(pathfile)
    tar.close()

    
