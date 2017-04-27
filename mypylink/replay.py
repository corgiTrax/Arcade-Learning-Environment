#!/usr/bin/env python
# Author: Zhuode Liu

# Replay the game using saved frames and gaze positions recorded in a ASC file 
# (ASC files are converted from EDF files, which are produced by EyeLink eyetracker)

import sys, pygame, time, os, re, tarfile, cStringIO as StringIO
from pygame.constants import *

def frameid_from_filename(fname): 
    """ Extract '23' from '0_blahblah/23.png' """

    a, b = os.path.splitext(os.path.basename(fname))
    try:
        frameid = int(a)
    except ValueError as ex:
        raise ValueError("cannot convert filename '%s' to frame ID (an integer)" % fname)
    return frameid

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
        while prev_idx + 1 != frameid_from_filename(fname):
            print ("Warning: there is a gap between frames. Missing frame ID: %d" % (prev_idx+1))
            hasWarning = True
            prev_idx += 1
        prev_idx += 1

    if hasWarning:
        print "There are warnings. Sleeping for 2 sec..."
        time.sleep(2)

    return png_files

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

    return frameid2pos

class drawgc_wrapper:
    def __init__(self):
        self.cursor = pygame.image.load('shooting-game-target.png')
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
        print "Usage: %s saved_frames_png_tar asc_file" % (sys.argv[0])
        sys.exit(0)

    tar = tarfile.open(sys.argv[1], 'r')
    asc_path = sys.argv[2]

    png_files = tar.getnames()
    png_files = preprocess_and_sanity_check(png_files)
    print "\nYou can control the replay using keyboard. Try pressing space/up/down/left/right." 
    print "For all available keys, see event_handler_func() code.\n"
    print "Uncompressing PNG tar file into memory (/dev/shm/)..."
    tar.extractall("/dev/shm/")

    # init pygame and other stuffs
    xSCALE, ySCALE = 6, 3
    w, h = 160*xSCALE, 210*ySCALE
    pygame.init()
    pygame.display.set_mode((w, h), RESIZABLE | DOUBLEBUF | RLEACCEL, 32)
    screen = pygame.display.get_surface()
    print "Reading gaze data in ASC file into memory..."
    frameid2pos = read_gaze_data_asc_file(asc_path)
    dw = drawgc_wrapper()

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
        s = pygame.image.load("/dev/shm/" + png_files[ds.cur_frame_id])
        s = pygame.transform.scale(s, (w,h))
        screen.blit(s, (0,0))
        if ds.cur_frame_id in frameid2pos and len(frameid2pos[ds.cur_frame_id])>0:
            for gaze_pos in frameid2pos[ds.cur_frame_id]:
                dw.draw_gc(screen, gaze_pos)
                if not ds.draw_many_gazes: break
        else:
            print "Warning: No gaze data for frame ID %d" % ds.cur_frame_id
        pygame.display.flip()

        if not ds.pause:
            ds.cur_frame_id += 1

    print "Replay ended."

# The method used in this script to align a gaze position to each frame is:
# For frame #i, attach the gaze position immediately before frame #i+1.
# That is, search for the last gaze position immediately before "MSG <timestamp> SCR_RECORDER FRAMEID i+1",
# An example having only 3 frames is given below

'''
MSG     472750 SCR_RECORDER FRAMEID 1
472750    485.3   305.9  1020.0 ...
472751    485.6   306.7  1019.0 ...
472752    485.9   307.8  1018.0 ...
472753    486.1   308.8  1017.0 ...
472754    486.6   309.3  1020.0 ...
472755    486.9   309.3  1023.0 ...
472756    486.0   309.4  1028.0 ...
472757    484.8   309.6  1027.0 ...
472758    483.6   309.9  1027.0 ...
472759    483.6   310.1  1025.0 ...
472760    483.6   310.0  1025.0 ...
472761    483.6   310.0  1027.0 ...
472762    483.6   309.9  1027.0 ...
472763    483.7   309.9  1027.0 ...
472764    483.7   309.9  1026.0 ...
472765    483.7   309.5  1027.0 ...
472766    483.7   309.1  1027.0 ...
472767    483.7   308.5  1027.0 ...
472768    483.7   308.1  1026.0 ...
472769    484.1   307.6  1026.0 ...
472770    484.5   307.4  1026.0 ...
472771    485.0   307.5  1026.0 ...
472772    485.2   307.8  1024.0 ...
472773    485.4   307.5  1023.0 ...
472774    485.4   307.1  1029.0 ...
472775    485.3   306.7  1035.0 ...
472776    485.1   306.7  1038.0 ...
472777    484.9   307.3  1035.0 ...
472778    484.8   307.7  1032.0 ...
472779    484.7   308.1  1032.0 ...
472780    484.6   308.1  1032.0 ...
472781    484.3   308.2  1036.0 ...
472782    484.1   308.3  1040.0 ...   <----- this gaze position gets attached to frame 1
MSG     472783 SCR_RECORDER FRAMEID 2
472783    484.2   308.4  1044.0 ...
472784    484.4   308.1  1041.0 ...
472785    484.7   307.8  1037.0 ...
472786    484.7   307.3  1034.0 ...
472787    485.1   307.0  1034.0 ...
472788    485.5   307.0  1035.0 ...
472789    485.7   307.7  1036.0 ...
472790    485.5   308.4  1037.0 ...
472791    485.1   308.9  1039.0 ...
472792    484.9   309.0  1040.0 ...
472793    484.7   308.7  1041.0 ...
472794    484.6   308.4  1041.0 ...
472795    484.5   308.6  1040.0 ...
472796    485.0   309.5  1040.0 ...
472797    485.6   310.3  1040.0 ...
472798    485.2   310.1  1041.0 ...
472799    484.3   308.1  1041.0 ...
472800    484.0   306.2  1041.0 ...
472801    484.8   304.6  1041.0 ...
472802    485.8   304.5  1042.0 ...
472803    486.0   304.9  1043.0 ...
472804    486.2   305.8  1046.0 ...
472805    485.4   307.2  1048.0 ...
472806    484.7   308.2  1047.0 ...
472807    483.7   308.8  1044.0 ...
472808    483.3   309.2  1042.0 ...
472809    482.8   309.6  1041.0 ...
472810    482.7   309.0  1041.0 ...
472811    482.9   308.0  1044.0 ...
472812    483.3   307.8  1048.0 ...
472813    483.6   308.5  1052.0 ...
472814    484.0   309.3  1050.0 ...  <----- this position gets attached to frame 2 
MSG     472815 SCR_RECORDER FRAMEID 3 <----- Frame 3, the last frame, has no gaze position attached to it
472815    484.2   309.2  1048.0 ...
472816    484.4   309.0  1047.0 ...
472817    484.4   309.0  1049.0 ...
472818    483.8   309.0  1050.0 ...
472819    483.1   309.6  1050.0 ...
'''
