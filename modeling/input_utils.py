import os, re, threading, time, sys
import numpy as np
from IPython import embed
from scipy import misc
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
import base_input_utils as BIU
import vip_constants as V

# This method is for the expr that multiply the game frame using a predicted gaze heatmap, 
# which is the output from another model that predicts gaze.
# So make sure you do the check for yourself and provide this method correct data.
def load_predicted_gaze_heatmap_into_dataset_train_GHmap_val_GHmap(train_npz, val_npz, d, pastK):
    train_npz = np.load(train_npz)
    val_npz = np.load(val_npz)
    d.train_GHmap = train_npz['heatmap']
    d.val_GHmap = val_npz['heatmap']
    # npz file from pastK models has pastK-fewer data, so we need to know use value of pastK
    d.train_imgs = d.train_imgs[pastK:]
    d.val_imgs = d.val_imgs[pastK:]
    d.train_fid = d.train_fid[pastK:]
    d.val_fid = d.val_fid[pastK:]
    d.train_weight = d.train_weight[pastK:]
    d.val_weight = d.val_weight[pastK:]
    d.train_lbl = d.train_lbl[pastK:]
    d.val_lbl = d.val_lbl[pastK:]

    def validate_data(npz_fid, dataset_fid, imgs, GHmap):
        assert imgs.shape[0] == GHmap.shape[0], \
            "the number of image data does not match the number of gaze heat map data: %d vs %d" % (imgs.shape[0], GHmap.shape[0])
        for i in range(len(npz_fid)):
            assert tuple(npz_fid[i]) == tuple(dataset_fid[i]), \
                "fid in dataset and fid in npz file does not match: npz_fid[%d]=%s, dataset_fid[%d]=%s" % (i,str(npz_fid[i]),i,str(dataset_fid[i]))
    validate_data(train_npz['fid'], d.train_fid, d.train_imgs, d.train_GHmap)
    validate_data(val_npz['fid'], d.val_fid, d.val_imgs, d.val_GHmap)

def read_gaze_data_asc_file_2(fname):
    """
    Reads a ASC file and returns the following:
    all_gaze (Type:list) A tuple (utid, t, x, y) of all gaze messages in the file.
        where utid is the trial ID to which this gaze belongs. 
        It is extracted from the UTID of the most recent frame
    all_frame (Type:list) A tuple (f_time, frameid, UTID) of all frame messages in the file.
        where f_time is the time of the message. frameid is the unique ID used across the whole project.
    frameid2action (Type: dict) a dictionary mapping frame ID to action label
    (This function is designed to correctly handle the case that the file might be the concatenation of multiple asc files)
    """

    with open(fname, 'r') as f:
        lines = f.readlines()
    frameid, xpos, ypos = "BEFORE-FIRST-FRAME", None, None
    cur_frame_utid = "BEFORE-FIRST-FRAME"
    frameid2action = {frameid: None}
    all_gaze = []
    all_frame = []
    scr_msg = re.compile("MSG\s+(\d+)\s+SCR_RECORDER FRAMEID (\d+) UTID (\w+)")
    freg = "[-+]?[0-9]*\.?[0-9]+" # regex for floating point numbers
    gaze_msg = re.compile("(\d+)\s+(%s)\s+(%s)" % (freg, freg))
    act_msg = re.compile("MSG\s+(\d+)\s+key_pressed atari_action (\d+)")

    for (i,line) in enumerate(lines):
        
        match_sample = gaze_msg.match(line)
        if match_sample:
            g_time, xpos, ypos = int(match_sample.group(1)), match_sample.group(2), match_sample.group(3)
            g_time, xpos, ypos = int(g_time), float(xpos), float(ypos)
            all_gaze.append((cur_frame_utid, g_time, xpos, ypos))
            continue

        match_scr_msg = scr_msg.match(line)
        if match_scr_msg: # when a new id is encountered
            f_time, frameid, UTID = match_scr_msg.group(1), match_scr_msg.group(2), match_scr_msg.group(3)
            f_time = int(f_time)
            cur_frame_utid = UTID
            frameid = BIU.make_unique_frame_id(UTID, frameid)
            all_frame.append((f_time, frameid, UTID))
            frameid2action[frameid] = None
            continue

        match_action = act_msg.match(line)
        if match_action:
            timestamp, action_label = match_action.group(1), match_action.group(2)
            if frameid2action[frameid] is None:
                frameid2action[frameid] = int(action_label)
            else:
                print "Warning: there are more than 1 action for frame id %s. Not supposed to happen." % str(frameid)
            continue

    if len(all_frame) < 1000: # simple sanity check
        print "Warning: did you provide the correct ASC file? Because the data for only %d frames is detected" % (len(frameid2pos))
        raw_input("Press any key to continue")

    return all_gaze, all_frame, frameid2action


# Note(Zhuode Liu): In this dataset, all gaze positions belonging to the current frame is converted
# into a gaze heat map and is applied BIU.preprocess_gaze_heatmap() which, at the time of writing, does gaussian blur.
class DatasetWithGaze(BIU.Dataset):
  frameid2pos, frameid2heatmap, frameid2action_notused = None, None, None
  train_GHmap, val_GHmap = None, None # GHmap means gaze heap map
  
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, GAZE_POS_ASC_FILE, bg_prob_density, gaussian_sigma):
    super(DatasetWithGaze, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)
    print "Reading gaze data ASC file, and converting per-frame gaze positions to heat map..."
    self.frameid2pos, self.frameid2action_notused, _ = BIU.read_gaze_data_asc_file(GAZE_POS_ASC_FILE)
    self.train_GHmap = np.zeros([self.train_size, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1], dtype=np.float32)
    self.val_GHmap = np.zeros([self.val_size, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1], dtype=np.float32)

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

    sigmaH = gaussian_sigma * RESIZE_SHAPE[0] / 210.0
    sigmaW = gaussian_sigma * RESIZE_SHAPE[1] / 160.0

    self.train_GHmap = BIU.preprocess_gaze_heatmap(self.train_GHmap, sigmaH, sigmaW, bg_prob_density)
    self.val_GHmap = BIU.preprocess_gaze_heatmap(self.val_GHmap, sigmaH, sigmaW, bg_prob_density)
    print "Done. BIU.convert_gaze_pos_to_heap_map() and convolution used: %.1fs" % (time.time()-t1)


# Note(Zhuode Liu): This class represents a historical model that doesn't perform well.
# It generates past K frames like class Dataset_PastKFrames does, but what's different is that
# it first search for the frame that is at least `ms_before` (the input argument). Starting from
# there, it extracts past K frame. So if ms_before==0, it behaves the same as Dataset_PastKFrames.
class Dataset_PastKFramesByTime(BIU.Dataset):
  def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, GAZE_POS_ASC_FILE, K, stride=1, ms_before=0):
    super(Dataset_PastKFramesByTime, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)
    self.train_imgs_bak, self.val_imgs_bak = self.train_imgs, self.val_imgs

    def data_is_sorted_by_timestamp(fid_list):
        # It does (1) stable sort (2) ignore and only compare the second key (i.e.frame number)
        sorted_fid_list = sorted(fid_list,cmp=lambda x,y:int(x[0]==y[0])*(x[1]-y[1])) 
        return sorted_fid_list == fid_list
    assert data_is_sorted_by_timestamp(self.train_fid)
    assert data_is_sorted_by_timestamp(self.val_fid)
    # Since later code extracts adjacant indice of the data via something like img[100:104], "data_is_sorted_by_timestamp" must be true.
    # (self.train/val_fid is a list that stores the corresponding frame ID of self.train/val_imgs)
    # If assertion failed, rewrite the dataset generation python file to satify this assumption

    t1=time.time()
    print "Reading gaze data ASC file..."
    _, all_frame, frameid2action = read_gaze_data_asc_file_2(GAZE_POS_ASC_FILE)
    print "Done."
    self.train_imgs, self.train_lbl = transform_to_past_K_frames_ByTime(self.train_imgs, self.train_fid, all_frame, frameid2action, K, stride, ms_before)
    self.val_imgs, self.val_lbl = transform_to_past_K_frames_ByTime(self.val_imgs, self.val_fid, all_frame, frameid2action, K, stride, ms_before)
    print "Time spent to transform train/val data to pask K frames: %.1fs" % (time.time()-t1)

def transform_to_past_K_frames_ByTime(frame_dataset, frame_fid_list, all_frame, frameid2action, K, stride, ms_before):
    newdat = []
    newlbl = []
    frameid2time = {frameid:f_time for (f_time, frameid, UTID) in all_frame}
    time = [frameid2time[fid] for fid in frame_fid_list] # find the timestamp for each frame in frame_dataset

    l, r = 0, None # two pointers: left, right
    for r in range(K*stride, len(frame_dataset)):

        # move l to the correct position
        if frame_fid_list[l][0] != frame_fid_list[r][0]: 
            l=r # [0] is UTID. If not equal, l and r are in different trials; so time[l] and time[r] is incomparable; so we need to set l=r.
        while l<r and time[r] - time[l+1] >= ms_before: 
            l+=1

        leftmost = l-K*stride # do some check to make the line "cur = frame_dataset[l : l-K*stride : -stride]" below work correctly
        if leftmost<0 or frame_fid_list[leftmost][0] != frame_fid_list[r][0]:  
        # the leftmost idx is not a valid idx, or is in a different trial than current frame (the frame with idx r)
            continue

        # transform the shape (K, 84, 84, CH) into (84, 84, CH*K)
        cur = frame_dataset[l : l-K*stride : -stride] # using "-stride" instead of "stride" lets the indexing include l rather than exclude l
        cur = cur.transpose([1,2,3,0])
        cur = cur.reshape(cur.shape[0:2]+(-1,))

        newdat.append(cur)
        newlbl.append(frameid2action[frame_fid_list[r]])
        if len(newdat)>1: assert (newdat[-1].shape == newdat[-2].shape) # simple sanity check
    newdat_np = np.array(newdat)
    newlbl_np = np.array(newlbl, dtype=np.int32)
    return newdat_np, newlbl_np

# Note(Zhuode Liu): This class represents a historical model that doesn't perform well.
# It generates the gaze heat map using gazes within a given *time window* instead of past K frames.
# Later on we thought about why it performs badly: the gaze in (say) 100ms ago is for the object 100ms ago,
# but that object could be moving to a different location in current frame. So the gaze heat map doesn't overlap
# with current frame. However, this method still has research value though. So we leave it here for future modification.
class DatasetWithGazeWindow(BIU.Dataset):
    def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, GAZE_POS_ASC_FILE, bg_prob_density, gaussian_sigma,
        window_left_bound_ms=1000, window_right_bound_ms=0):
      super(DatasetWithGazeWindow, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)
      self.RESIZE_SHAPE = RESIZE_SHAPE
      self.bg_prob_density = bg_prob_density
      self.gaussian_sigma=gaussian_sigma
      self.window_left_bound, self.window_right_bound = window_left_bound_ms, window_right_bound_ms
      print "Reading gaze data ASC file, and converting per-frame gaze positions to heat map..."
      all_gaze, all_frame, _ = read_gaze_data_asc_file_2(GAZE_POS_ASC_FILE)
      self.rescale_and_clip_gaze_pos_on_all_gaze(all_gaze)
      self.train_GHmap = np.empty([self.train_size, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1], dtype=np.float32)
      self.val_GHmap = np.empty([self.val_size, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1], dtype=np.float32)
      print "Running convert_gaze_data_to_map()..."
      t1=time.time()
      self.frameid2GH, self.frameid2gazetuple = self.convert_gaze_data_to_heat_map_proprietary(all_gaze, all_frame)
      self.prepare_train_val_gaze_data()
      print "Done. convert_gaze_data_to_heat_map() and convolution used: %.1fs" % (time.time()-t1)
      
    def prepare_train_val_gaze_data(self):
        print "Assign a heap map for each frame in train and val dataset..."
        for (i,fid) in enumerate(self.train_fid):
            self.train_GHmap[i] = self.frameid2GH[fid]
        for (i,fid) in enumerate(self.val_fid):
            self.val_GHmap[i] = self.frameid2GH[fid]
        print "Applying Gaussian Filter and normalization on train/val gaze heat map..."
        sigmaH = self.gaussian_sigma * self.RESIZE_SHAPE[0] / 210.0
        sigmaW = self.gaussian_sigma * self.RESIZE_SHAPE[1] / 160.0
        self.train_GHmap = BIU.preprocess_gaze_heatmap(self.train_GHmap, sigmaH, sigmaW, self.bg_prob_density)
        self.val_GHmap = BIU.preprocess_gaze_heatmap(self.val_GHmap, sigmaH, sigmaW, self.bg_prob_density)

    def rescale_and_clip_gaze_pos_on_all_gaze(self, all_gaze):
        bad_count=0
        for (i, (utid, t, x, y)) in enumerate(all_gaze):
            isbad, newx, newy = BIU.rescale_and_clip_gaze_pos(x,y,self.RESIZE_SHAPE[0],self.RESIZE_SHAPE[1])
            bad_count += isbad
            all_gaze[i] = (utid, t, newx, newy)

        print "Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/len(all_gaze), len(all_gaze))
        print "'Bad' means the gaze position is outside the 160*210 screen"

    def convert_gaze_data_to_heat_map_proprietary(self, all_gaze, all_frame):
        GH = np.zeros([self.RESIZE_SHAPE[0], self.RESIZE_SHAPE[1], 1], dtype=np.float32)
        left, right = 0,0
        frameid2GH = {}
        frameid2gazetuple={}

        def left_matched(utid, frame_utid, gaze_t, frame_t, window_left_bound):
            if frame_t  - gaze_t > window_left_bound: return False
            elif utid != frame_utid: return False
            else: return True

        def right_matched(utid, frame_utid, gaze_t, frame_t, window_right_bound):
            if frame_t - gaze_t > window_right_bound: return False
            elif utid != frame_utid: return False
            else: return True

        for (frame_t, frameid, frame_utid) in all_frame:

            while right<len(all_gaze):
                utid, gaze_t, x, y = all_gaze[right] 
                if right_matched(utid, frame_utid, gaze_t, frame_t, self.window_right_bound): break
                GH[y,x,0] += 1.0 
                right+=1 

            while left<right:
                utid, gaze_t, x, y = all_gaze[left]
                if left_matched(utid, frame_utid, gaze_t, frame_t, self.window_left_bound): break
                GH[y,x,0] -= 1.0
                left+=1 

            gaze_for_cur_frame = all_gaze[left:right]
            if gaze_for_cur_frame: # non-empty
                assert gaze_for_cur_frame[-1][1] < frame_t # simple sanity-check: the timestamp of the latest gaze position should be smaller than the timestamp of current frame

            frameid2GH[frameid] = GH.copy()
            frameid2gazetuple[frameid] = gaze_for_cur_frame

        return frameid2GH, frameid2gazetuple