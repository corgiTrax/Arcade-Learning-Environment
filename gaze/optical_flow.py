import cv2, tarfile, time, sys, os
import numpy as np
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
from base_input_utils import frameid_from_filename, ForkJoiner
import matplotlib.pyplot as plt
from replay import preprocess_and_sanity_check

RESIZE_SHAPE = (84,84)
INPUT_SHAPE = (210,160)

if len(sys.argv) < 2:
   print "Usage: %s saved_frames_png_tar" % (sys.argv[0])
   sys.exit(0)

dataset = sys.argv[1]
dataset_name = dataset.split('.')[-3].split('/')[-1] # eg: 42_RZ_4988291_May-16-21-33-46
save_path = '/scratch/cluster/zharucs/data/optical_flow/'
if not os.path.exists(save_path + dataset_name):
    os.mkdir(save_path + dataset_name)

tar = tarfile.open(dataset, 'r')
png_files = tar.getnames()
#tar.extractall("../../dataset_gaze")
png_files = preprocess_and_sanity_check(png_files)

num_thread = 6
l = len(png_files)
def read_thread(PID):
    for i in range(PID+1, l, num_thread):
        prev = cv2.imread('/scratch/cluster/zharucs/data/' + png_files[i-1], 0)
        cur = cv2.imread('/scratch/cluster/zharucs/data/' + png_files[i], 0)

        # calculate optical flow. 
        # The function is: calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # we only use the information of magnitude
        fx, fy = flow[:,:,0], flow[:,:,1]
        # ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        
        gray = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.resize(gray, RESIZE_SHAPE)
        cv2.imwrite(save_path + png_files[i], gray)

        # print the processing bar
        print "\r%d/%d" % (i,l),
        sys.stdout.flush()

print "Saving optical flow images for dataset %s" % dataset_name
t1 = time.time()
o=ForkJoiner(num_thread=num_thread, target=read_thread)
o.join()
print "Done. Saving optical flow images takes %.1fs" % (time.time()-t1)
