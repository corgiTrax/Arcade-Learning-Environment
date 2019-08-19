#!/usr/bin/env python
import os, re, threading, time
import numpy as np
from IPython import embed
from scipy import misc
import vip_constants as V

# Don't import tensorflow or keras here, as people who import this file might not have keras or tf

class Dataset(object):
  train_imgs, train_lbl, train_fid, train_size, train_weight = None, None, None, None, None
  val_imgs, val_lbl, val_fid, val_size, val_weight = None, None, None, None, None
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE):
    t1=time.time()
    print ("Reading all training data into memory...")
    self.train_imgs, self.train_lbl, self.train_gaze, self.train_fid, self.train_weight = read_np_parallel(LABELS_FILE_TRAIN, RESIZE_SHAPE)
    self.train_size = len(self.train_lbl)
    print ("Reading all validation data into memory...")
    self.val_imgs, self.val_lbl, self.val_gaze, self.val_fid, self.val_weight = read_np_parallel(LABELS_FILE_VAL, RESIZE_SHAPE)
    self.val_size = len(self.val_lbl)
    print ("Time spent to read train/val data: %.1fs" % (time.time()-t1))

    from collections import defaultdict
    d = defaultdict(int)
    for lbl in self.val_lbl: d[lbl] += 1
    print ("Baseline Accuracy (predicting to the majority class label): %.1f%%" % (100.0*max(d.values())/len(self.val_lbl)))

    print ("Performing standardization (x-mean)...")
    self.standardize()
    print ("Done.")

  def standardize(self):
    self.mean = np.mean(self.train_imgs, axis=(0,1,2))
    self.train_imgs -= self.mean # done in-place --- "x-=mean" is faster than "x=x-mean"
    self.val_imgs -= self.mean

  def convert_one_hot_label_to_prob_dist(self, moving_avarage_window_radius, NUM_CLASSES):
    self.train_lbl = self._convert_one_hot_label_to_prob_dist(self.train_lbl, moving_avarage_window_radius, NUM_CLASSES)
    self.val_lbl = self._convert_one_hot_label_to_prob_dist(self.val_lbl, moving_avarage_window_radius, NUM_CLASSES)
  def _convert_one_hot_label_to_prob_dist(self, lbl, r, NUM_CLASSES):
    result = np.zeros((len(lbl), NUM_CLASSES), dtype=np.float16)
    for i in range(len(lbl)):
        result[i][lbl[i]] = 1.0
    for i in range(len(lbl)):
        left, right = max(0, i-r), min(len(lbl), i+r+1) # +1 because python indexing like arr[A:B] excludes arr[B]
        result[i] = np.mean(result[left:right], axis=0)
    return result

class Dataset_PastKFrames(Dataset):
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, K, stride=1, before=0):
    super(Dataset_PastKFrames, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)
    self.train_imgs_bak, self.val_imgs_bak = self.train_imgs, self.val_imgs

    t1=time.time()
    self.train_imgs = transform_to_past_K_frames(self.train_imgs, K, stride, before)
    self.val_imgs = transform_to_past_K_frames(self.val_imgs, K, stride, before)
    # Trim labels. This is assuming the labels align with the training examples from the back!!
    # Could cause the model unable to train  if this assumption does not hold
    self.train_lbl = self.train_lbl[-self.train_imgs.shape[0]:]
    self.val_lbl = self.val_lbl[-self.val_imgs.shape[0]:]
    print ("Time spent to transform train/val data to pask K frames: %.1fs" % (time.time()-t1))

def transform_to_past_K_frames(original, K, stride, before):
    newdat = []
    for i in range(before+K*stride, len(original)):
        # transform the shape (K, H, W, CH) into (H, W, CH*K)
        cur = original[i-before : i-before-K*stride : -stride] # using "-stride" instead of "stride" lets the indexing include i rather than exclude i
        cur = cur.transpose([1,2,3,0])
        cur = cur.reshape(cur.shape[0:2]+(-1,))
        newdat.append(cur)
        if len(newdat)>1: assert (newdat[-1].shape == newdat[-2].shape) # simple sanity check
    newdat_np = np.array(newdat)
    return newdat_np

def frameid_from_filename(fname): 
    """ Extract 'exprname','randnum','23' from '0_blahblah/exprname_randnum_23.png' """

    a, b = os.path.splitext(os.path.basename(fname))
    try:
        exprname, randnum, frameid = a.split('_')
        UTID = exprname + '_' + randnum   # This the definition of an "unique trail id"
    except ValueError as ex:
        raise ValueError("cannot convert filename '%s' to frame ID" % fname)
    return make_unique_frame_id(UTID, frameid)

def make_unique_frame_id(UTID, frameid):
    return (hash(UTID), int(frameid))

def read_gaze_data_asc_file(fname):
    """ This function reads a ASC file and returns 
        a dictionary mapping frame ID to a list of gaze positions,
        a dictionary mapping frame ID to action """

    with open(fname, 'r') as f:
        lines = f.readlines()
    frameid, xpos, ypos = "BEFORE-FIRST-FRAME", None, None
    frameid2pos = {frameid: []}
    frameid2action = {frameid: None}
    frameid2duration = {frameid: None} 
    frameid2unclipped_reward = {frameid: None}
    frameid2episode = {frameid: None}
    start_timestamp = 0
    scr_msg = re.compile(r"MSG\s+(\d+)\s+SCR_RECORDER FRAMEID (\d+) UTID (\w+)")
    freg = r"[-+]?[0-9]*\.?[0-9]+" # regex for floating point numbers
    gaze_msg = re.compile(r"(\d+)\s+(%s)\s+(%s)" % (freg, freg))
    act_msg = re.compile(r"MSG\s+(\d+)\s+key_pressed atari_action (\d+)")    
    reward_msg = re.compile(r"MSG\s+(\d+)\s+reward (\d+)")
    episode_msg = re.compile(r"MSG\s+(\d+)\s+episode (\d+)")

    for (i,line) in enumerate(lines):
        match_sample = gaze_msg.match(line)
        if match_sample:
            timestamp, xpos, ypos = match_sample.group(1), match_sample.group(2), match_sample.group(3)
            xpos, ypos = float(xpos), float(ypos)
            frameid2pos[frameid].append((xpos,ypos))
            continue

        match_scr_msg = scr_msg.match(line)
        if match_scr_msg: # when a new id is encountered
            old_frameid = frameid 
            timestamp, frameid, UTID = match_scr_msg.group(1), match_scr_msg.group(2), match_scr_msg.group(3)
            frameid2duration[old_frameid] = int(timestamp) - start_timestamp 
            start_timestamp = int(timestamp)
            frameid = make_unique_frame_id(UTID, frameid)
            frameid2pos[frameid] = []
            frameid2action[frameid] = None
            continue

        match_action = act_msg.match(line)
        if match_action:
            timestamp, action_label = match_action.group(1), match_action.group(2)
            if frameid2action[frameid] is None:
                frameid2action[frameid] = int(action_label)
            else:
                print ("Warning: there is more than 1 action for frame id %s. Not supposed to happen." % str(frameid))
            continue

        match_reward = reward_msg.match(line)
        if match_reward:
            timestamp, reward = match_reward.group(1), match_reward.group(2)
            if frameid not in frameid2unclipped_reward:
                frameid2unclipped_reward[frameid] = int(reward)
            else:
                print ("Warning: there is more than 1 reward for frame id %s. Not supposed to happen." % str(frameid))
            continue

        match_episode = episode_msg.match(line)
        if match_episode:
            timestamp, episode = match_episode.group(1), match_episode.group(2)
            assert frameid not in frameid2episode, "ERROR: there is more than 1 episode for frame id %s. Not supposed to happen." % str(frameid)
            frameid2episode[frameid] = int(episode) 
            continue

    frameid2pos[frameid] = [] # throw out gazes after the last frame, because the game has ended but eye tracker keeps recording

    if len(frameid2pos) < 1000: # simple sanity check
        print ("Warning: did you provide the correct ASC file? Because the data for only %d frames is detected" % (len(frameid2pos)))
        raw_input("Press any key to continue")

    few_cnt = 0
    for v in frameid2pos.values():
        if len(v) < 10: few_cnt += 1
    print ("Warning:  %d frames have less than 10 gaze samples. (%.1f%%, total frame: %d)" % \
            (few_cnt, 100.0*few_cnt/len(frameid2pos), len(frameid2pos)))
    return frameid2pos, frameid2action, frameid2duration, frameid2unclipped_reward, frameid2episode


def read_np_parallel(label_file, RESIZE_SHAPE, num_thread=6, preprocess_deprecated=True):
    """
    Args:
        preprocess_deprecated: This is newly added for backward-compatibility. Old code assume this funciton does
            some resizeing and scaling. But now we defer this for easier intergration with OpenAI's repo "baselines".

    Read the whole dataset into memory. 
    Provide a label file (text file) which has "{image_path} {label}\n" per line.
    Returns a numpy array of the images, and a numpy array of labels
    """
    labels, fids = [], []
    png_files = []
    gaze = [] # TODO:  see TODO below
    weight = []
    with open(label_file,'r') as f:
        for line in f:
            line=line.strip()
            if line.startswith("#") or line == "": 
                continue # skip comments or empty lines
            fname, lbl, x, y, w = line.split(' ') # TODO: Luxin adds the last gaze positions (x,y) into label file.
            # TODO This is not a good practice. We should delete the code that reads them here, and delete the code 
            # TODO that writes them in create_train_val_dataset.py. And then find the code which needs this information
            # to get the information from "frameid2pos"(a dict returned by read_gaze_data_asc_file()) instead.
            png_files.append(fname)
            labels.append(int(lbl))
            fids.append(frameid_from_filename(fname))
            gaze.append((float(x)*RESIZE_SHAPE[1]/V.SCR_W, float(y)*RESIZE_SHAPE[0]/V.SCR_H))
            weight.append(float(w))
    N = len(labels)
    imgs = [None] * N
    labels = np.asarray(labels, dtype=np.int32)
    gaze = np.asarray(gaze, dtype=np.float16)
    weight = np.asarray(weight, dtype=np.float16)

    def read_thread(PID):
        d = os.path.dirname(label_file)
        for i in range(PID, N, num_thread):
            if preprocess_deprecated:
                img = misc.imread(os.path.join(d, png_files[i]), 'Y') # 'Y': grayscale  
                img = misc.imresize(img, [RESIZE_SHAPE[0],RESIZE_SHAPE[1]], interp='bilinear')
                img = np.expand_dims(img, axis=2)
                img = img.astype(np.float16) / 255.0 # normalize image to [0,1]
            else:
                img = misc.imread(os.path.join(d, png_files[i])) # uint8 RGB (210,160,3)
            imgs[i] = img

    o=ForkJoiner(num_thread=num_thread, target=read_thread)
    o.join()
    return np.asarray(imgs), labels, gaze, fids, weight


class ForkJoiner():
    def __init__(self, num_thread, target):
        self.num_thread = num_thread
        self.threads = [threading.Thread(target=target, args=[PID]) for PID in range(num_thread)]
        for t in self.threads: 
            t.start()
    def join(self):
        for t in self.threads: t.join()


def rescale_and_clip_gaze_pos(x,y,RESIZE_H,RESIZE_W):
    isbad=0
    newy, newx = int(y/V.SCR_H*RESIZE_H), int(x/V.SCR_W*RESIZE_W)
    if newx >= RESIZE_W or newx<0:
        isbad = 1
        newx = np.clip(newx, 0, RESIZE_W-1)
    if newy >= RESIZE_H or newy<0:
        isbad = 1
        newy = np.clip(newy, 0, RESIZE_H-1)
    return isbad, newx, newy

def convert_gaze_pos_to_heap_map(gaze_pos_list, out):
    h,w = out.shape[0], out.shape[1]
    bad_count = 0
    for (x,y) in gaze_pos_list: 
        try:
            out[int(y/V.SCR_H*h), int(x/V.SCR_W*w)] += 1
        except IndexError: # the computed X,Y position is not in the gaze heat map
            bad_count += 1
    return bad_count


# bg_prob_density seems to hurt accuracy. Better set it to 0
def preprocess_gaze_heatmap(GHmap, sigmaH, sigmaW, bg_prob_density, debug_plot_result=False):
    from scipy.stats import multivariate_normal
    import tensorflow as tf, keras as K # don't move this to the top, as people who import this file might not have keras or tf

    model = K.models.Sequential()

    model.add(K.layers.Lambda(lambda x: x+bg_prob_density, input_shape=(GHmap.shape[1],GHmap.shape[2],1)))

    if sigmaH > 0.0 and sigmaW > 0.0:
        lh, lw = int(4*sigmaH), int(4*sigmaW)
        x, y = np.mgrid[-lh:lh+1:1, -lw:lw+1:1] # so the kernel size is [lh*2+1,lw*2+1]
        pos = np.dstack((x, y))
        gkernel=multivariate_normal.pdf(pos,mean=[0,0],cov=[[sigmaH*sigmaH,0],[0,sigmaW*sigmaW]])
        assert gkernel.sum() > 0.95, "Simple sanity check: prob density should add up to nearly 1.0"

        model.add(K.layers.Lambda(lambda x: tf.pad(x,[(0,0),(lh,lh),(lw,lw),(0,0)],'REFLECT')))
        model.add(K.layers.Conv2D(1, kernel_size=gkernel.shape, strides=1, padding="valid", use_bias=False,
              activation="linear", kernel_initializer=K.initializers.Constant(gkernel)))
    else:
        print ("WARNING: Gaussian filter's sigma is 0, i.e. no blur.")
    # The following normalization hurts accuracy. I don't know why. But intuitively it should increase accuracy
    #def GH_normalization(x):
    #    sum_per_GH = tf.reduce_sum(x,axis=[1,2,3])
    #    sum_per_GH_correct_shape = tf.reshape(sum_per_GH, [tf.shape(sum_per_GH)[0],1,1,1])
    #    # normalize values to range [0,1], on a per heap-map basis
    #    x = x/sum_per_GH_correct_shape
    #    return x
    #model.add(K.layers.Lambda(lambda x: GH_normalization(x)))
    
    model.compile(optimizer='rmsprop', # not used
          loss='categorical_crossentropy', # not used
          metrics=None)
    output=model.predict(GHmap, batch_size=500)

    if debug_plot_result:
        print (r"""debug_plot_result is True. Entering IPython console. You can run:
                %matplotlib
                import matplotlib.pyplot as plt
                f, axarr = plt.subplots(1,2)
                axarr[0].imshow(gkernel)
                rnd=np.random.randint(output.shape[0]); print "rand idx:", rnd
                axarr[1].imshow(output[rnd,...,0])""")
        embed()
    
    shape_before, shape_after = GHmap.shape, output.shape
    assert shape_before == shape_after, """
    Simple sanity check: shape changed after preprocessing. 
    Your preprocessing code might be wrong. Check the shape of output tensor of your tensorflow code above"""
    return output

