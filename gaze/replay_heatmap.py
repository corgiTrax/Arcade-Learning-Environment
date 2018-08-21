#!/usr/bin/env python
# Author: Zhuode Liu

# Replay the game using saved frames and gaze positions recorded in a ASC file 
# Meanwhile overlay the predicted heatmaps on the frames
# (ASC files are converted from EDF files, which are produced by EyeLink eyetracker)
# This file takes two arguments:
#   the path and name of the dataset that you want to replay
#   the path of the directory where you save the predicted heatmaps (the directory named 'saliency')
#example: ./replay_heatmap.py ../../dataset_gaze/54_RZ_2461867_Aug-11-09-35-18 Image+OpticalFlow/Seaquest/45_pKf_dp0.4_k4s1/

import sys, pygame, time, os, re, tarfile, cStringIO as StringIO, numpy as np
from pygame.constants import *
from scipy import misc
import cv2
import input_utils as read_result_data
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
from base_input_utils import frameid_from_filename, read_gaze_data_asc_file
import vip_constants as V

RESIZE_SHAPE = (84,84)

def preprocess_and_sanity_check(png_files):
    hasWarning = False

    for fname in png_files:
        if not fname.endswith(".png"):
            print ("Warning: %s is not a PNG file. Deleting it from the frames list" % fname)
            hasWarning = True
            png_files.remove(fname)

    png_files = sorted(png_files, key=frameid_from_filename)

    prev_idx = 0
    for fname in png_files:
        # check if the index starts with one, and is free of gap (e.g. free of jumping from i to i+2)
        while prev_idx + 1 != frameid_from_filename(fname)[1]:
            print ("Warning: there is a gap between frames. Missing frame ID: %d" % (prev_idx+1))
            hasWarning = True
            prev_idx += 1
        prev_idx += 1

    if hasWarning:
        print "There are warnings. Sleeping for 2 sec..."
        time.sleep(2)

    return png_files

class drawgc_wrapper:
    def __init__(self, category):
        if category == 'original':
            self.cursor = pygame.image.load('target.png')
        elif category == 'predict':
            self.cursor = pygame.image.load('target_predict.png')
        self.cursorsize = (self.cursor.get_width(), self.cursor.get_height())

    def draw_gc(self, screen, gaze_position):
        '''draw the gaze-contingent window on screen '''
        region_topleft = (gaze_position[0] - self.cursorsize[0] // 2, gaze_position[1] - self.cursorsize[1] // 2)
        screen.blit(self.cursor, region_topleft) # Draws and shows the cursor content;

class DrawStatus:
    draw_many_gazes = True
    cur_frame_id = 1
    total_frame = None
    target_fps = 60
    pause = False
ds = DrawStatus()

def event_handler_func():
    global ds

    for event in pygame.event.get() :
      if event.type == pygame.KEYDOWN :
        if event.key == K_UP:
            print "Fast-backward 5 seconds"
            ds.cur_frame_id -= 5 * ds.target_fps
        elif event.key == K_DOWN:
            print "Fast-forward 5 seconds"
            ds.cur_frame_id += 5 * ds.target_fps
        if event.key == K_LEFT:
            print "Moving to previous frame"
            ds.cur_frame_id -= 1
        elif event.key == K_RIGHT:
            print "Moving to next frame"
            ds.cur_frame_id += 1
        elif event.key == K_F3:
            p = float(raw_input("Seeking through the video. Enter a percentage in float: "))
            ds.cur_frame_id = int(p/100*ds.total_frame)
        elif event.key == K_SPACE:
            ds.pause = not ds.pause
        elif event.key == K_F9:
            ds.draw_many_gazes = not ds.draw_many_gazes

            print "draw all gazes belonging to a frame: %s" % ("ON" if ds.draw_many_gazes else "OFF")
        elif event.key == K_F11:
            ds.target_fps -= 2
            print "Setting target FPS to %d" % ds.target_fps
        elif event.key == K_F12:
            ds.target_fps += 2
            print "Setting target FPS to %d" % ds.target_fps
    ds.cur_frame_id = max(0,min(ds.cur_frame_id, ds.total_frame))
    ds.target_fps = max(1, ds.target_fps)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: %s dataset_path_and_name model_path " % (sys.argv[0])
        sys.exit(0)

    # the two parameters to take
    DATASET_NAME = sys.argv[1] 
    MODEL = sys.argv[2]

    tar = tarfile.open(DATASET_NAME + '.tar.bz2', 'r')
    asc_path = DATASET_NAME + '.asc'
    dataset_dir = os.path.dirname(asc_path) + '/'
#    result_path = '../../dataset_gaze/' + RESULT_FILE_NAME + '.txt'
    
    png_files = tar.getnames()
    png_files = preprocess_and_sanity_check(png_files)
    print "\nYou can control the replay using keyboard. Try pressing space/up/down/left/right." 
    print "For all available keys, see event_handler_func() code.\n"
    print "Uncompressing PNG tar file into memory (/dev/shm/)..."
    UTIDhash = frameid_from_filename(png_files[2])[0]

    # init pygame and other stuffs
    RESIZE_SHAPE = (84,84,1)
    w, h = 160*V.xSCALE, 210*V.ySCALE
    pygame.init()
    pygame.display.set_mode((w, h), RESIZABLE | DOUBLEBUF | RLEACCEL, 32)
    screen = pygame.display.get_surface()
    print "Reading gaze data in ASC file into memory..."
    frameid2pos, _, _, _, _ = read_gaze_data_asc_file(asc_path)
    # print "Reading predict gaze positions into memory..."
    # predicts = read_result_data(result_path, RESIZE_SHAPE)
    dw = drawgc_wrapper('original')
    # dw_pred = drawgc_wrapper('predict')

    ds.target_fps = 60
    ds.total_frame = len(png_files)

    last_time = time.time()
    clock = pygame.time.Clock()
    while ds.cur_frame_id < ds.total_frame:
        clock.tick(ds.target_fps)  # control FPS 

        # Display FPS
        diff_time = time.time()-last_time
        if diff_time > 1.0:
            print 'FPS: %.1f Duration: %ds(%.1f%%)' % (clock.get_fps(), 
                ds.total_frame/ds.target_fps, 100.0*ds.cur_frame_id/ds.total_frame)
            last_time=time.time()

        event_handler_func()

        # Load PNG file and draw the frame and the gaze-contingent window
        img = cv2.imread(dataset_dir + png_files[ds.cur_frame_id-1], cv2.IMREAD_GRAYSCALE)
        img = np.swapaxes(img, 0, 1)
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
        s = pygame.surfarray.make_surface(img)
        s = pygame.transform.scale(s, (w,h))
        pygame.Surface.convert_alpha(s)
        s.set_alpha(100)
        try:
            heatmap = pygame.image.load(MODEL + "/saliency/" + png_files[ds.cur_frame_id-1])
            heatmap = pygame.transform.smoothscale(heatmap, (w,h))
        except pygame.error:
            heatmap = s
            print "Warning: no predicted heatmap for frame ID %d" % ds.cur_frame_id
                
        screen.blit(heatmap,(0,0))
        screen.blit(s, (0,0))

        UFID=(UTIDhash, ds.cur_frame_id) # Unique frame ID in 'frameid2pos' is defined as a tuple: (UTID's hash value, frame number)
        
        #  With the regression model result
        # if UFID in frameid2pos and len(frameid2pos[UFID])>0:
        #     if UFID in predicts:
        #         for gaze_pos in frameid2pos[UFID]:
        #             dw.draw_gc(screen, gaze_pos)
        #             dw_pred.draw_gc(screen, predicts[UFID])
        #             if not ds.draw_many_gazes: break
        #     else:
        #         print "Warning: No predict gaze data for frame ID %d" % ds.cur_frame_id
        #         print "Sleeping for 10 sec..."
        #         time.sleep(10)
        #         for gaze_pos in frameid2pos[UFID]:
        #             dw.draw_gc(screen, gaze_pos)
        #             if not ds.draw_many_gazes: break
        # else:
        #     print "Warning: No gaze data for frame ID %d" % ds.cur_frame_id

        #  Without the regression model result
        if UFID in frameid2pos and len(frameid2pos[UFID])>0:
            for gaze_pos in frameid2pos[UFID]:
                dw.draw_gc(screen, gaze_pos)
                if not ds.draw_many_gazes: break
        else:
            print "Warning: No gaze data for frame ID %d" % ds.cur_frame_id

        pygame.display.flip()

        if not ds.pause:
            ds.cur_frame_id += 1

    print "Replay ended."

