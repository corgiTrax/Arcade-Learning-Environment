class DatasetWithGazeWindow(Dataset):
    def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, GAZE_POS_ASC_FILE, bg_prob_density,
        window_left_bound_ms=1000, window_right_bound_ms=0):
      super(DatasetWithGazeWindow, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)
      self.RESIZE_SHAPE = RESIZE_SHAPE
      self.bg_prob_density = bg_prob_density
      self.window_left_bound, self.window_right_bound = window_left_bound_ms, window_right_bound_ms
      print "Reading gaze data ASC file, and converting per-frame gaze positions to heat map..."
      all_gaze, all_frame = self.read_gaze_data_asc_file_proprietary(GAZE_POS_ASC_FILE)
      print "Running rescale_and_clip_gaze_pos()..."
      self.rescale_and_clip_gaze_pos(all_gaze)
      self.train_GHmap = np.empty([self.train_size, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1], dtype=np.float32)
      self.val_GHmap = np.empty([self.val_size, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1], dtype=np.float32)
      print "Running convert_gaze_data_to_map()..."
      import time
      t1=time.time()
      self.frameid2GH, self.frameid2gazetuple = self.convert_gaze_data_to_heat_map_proprietary(all_gaze, all_frame)
      self.prepare_train_val_gaze_data()
      print "Done. Elapsed Time: ", time.time()-t1
      
    def prepare_train_val_gaze_data(self):
        print "Assign a heap map for each frame in train and val dataset..."
        for (i,fid) in enumerate(self.train_fid):
            self.train_GHmap[i] = self.frameid2GH[fid]
        for (i,fid) in enumerate(self.val_fid):
            self.val_GHmap[i] = self.frameid2GH[fid]
        print "Applying Gaussian Filter and normalization on train/val gaze heat map..."
        self.train_GHmap = preprocess_gaze_heatmap(self.train_GHmap, 10, 10)
        self.val_GHmap = preprocess_gaze_heatmap(self.val_GHmap, 10, 10)

    def read_gaze_data_asc_file_proprietary(self, fname):
        """
        Reads a ASC file and returns the following:
        all_gaze (Type:list) A tuple (utid, t, x, y) of all gaze messages in the file.
            where utid is the trial ID to which this gaze belongs. 
            It is extracted from the UTID of the most recent frame
        all_frame (Type:list) A tuple (f_time, frameid, UTID) of all frame messages in the file.
            where f_time is the time of the message. frameid is the unique ID used across the whole project.
        (This function is designed to correctly handle the case that the file might be the concatenation of multiple asc files)
        """

        with open(fname, 'r') as f:
            lines = f.readlines()
        frameid, xpos, ypos = "BEFORE-FIRST-FRAME", None, None
        cur_frame_utid = "BEFORE-FIRST-FRAME"
        all_gaze = []
        all_frame = []
        scr_msg = re.compile("MSG\s+(\d+)\s+SCR_RECORDER FRAMEID (\d+) UTID (\w+)")
        freg = "[-+]?[0-9]*\.?[0-9]+" # regex for floating point numbers
        gaze_msg = re.compile("(\d+)\s+(%s)\s+(%s)" % (freg, freg))

        for (i,line) in enumerate(lines):

            match_scr_msg = scr_msg.match(line)
            if match_scr_msg: # when a new id is encountered
                f_time, frameid, UTID = match_scr_msg.group(1), match_scr_msg.group(2), match_scr_msg.group(3)
                f_time = int(f_time)
                cur_frame_utid = UTID
                frameid = make_unique_frame_id(UTID, frameid)
                all_frame.append((f_time, frameid, UTID))
                continue
            
            match_sample = gaze_msg.match(line)
            if match_sample:
                g_time, xpos, ypos = int(match_sample.group(1)), match_sample.group(2), match_sample.group(3)
                g_time, xpos, ypos = int(g_time), float(xpos), float(ypos)
                all_gaze.append((cur_frame_utid, g_time, xpos, ypos))
                continue

        if len(all_frame) < 1000: # simple sanity check
            print "Warning: did you provide the correct ASC file? Because the data for only %d frames is detected" % (len(frameid2pos))
            raw_input("Press any key to continue")
        return all_gaze, all_frame # frameid to gaze heap map

    def rescale_and_clip_gaze_pos(self, all_gaze):
        bad_count=0
        h,w = self.RESIZE_SHAPE[0], self.RESIZE_SHAPE[1]
        for (i, (utid, t, x, y)) in enumerate(all_gaze):
            newy, newx = int(y/V.SCR_H*h), int(x/V.SCR_W*w)
            if newx >= w or newx<0:
                bad_count +=1
            if newy >= h or newy<0:
                bad_count +=1
            newx = np.clip(newx, 0, w-1)
            newy = np.clip(newy, 0, h-1)
            all_gaze[i] = (utid, t, newx, newy)

        print "Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/len(all_gaze), len(all_gaze))
        print "'Bad' means the gaze position is outside the 160*210 screen"

    def convert_gaze_data_to_heat_map_proprietary(self, all_gaze, all_frame):
        GH = np.full([self.RESIZE_SHAPE[0], self.RESIZE_SHAPE[1], 1], dtype=np.float32, fill_value=self.bg_prob_density)
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

