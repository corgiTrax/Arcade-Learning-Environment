#-------------------------------------------------------------------------------
# Name:        main
# Purpose:     Testing the package pySaliencyMap
#
# Author:      Akisato Kimura <akisato@ieee.org>
#
# Created:     May 4, 2014
# Copyright:   (c) Akisato Kimura 2014-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------

import cv2
import os, sys, tarfile, time
from bottom_up.pySaliencyMap import pySaliencyMap
from scipy import misc
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
from base_input_utils import frameid_from_filename, ForkJoiner
from replay import preprocess_and_sanity_check

if len(sys.argv) < 2:
   print "Usage: %s saved_frames_png_tar" % (sys.argv[0])
   sys.exit(0)

RESIZE_SHAPE = (84, 84)
dataset = sys.argv[1]
dataset_name = dataset.split('.')[-3].split('/')[-1] # eg: 42_RZ_4988291_May-16-21-33-46
save_path = '/scratch/cluster/zharucs/data/bottom_up/'
if not os.path.exists(save_path + dataset_name):
    os.mkdir(save_path + dataset_name)

tar = tarfile.open(dataset, 'r')
png_files = tar.getnames()
png_files = preprocess_and_sanity_check(png_files)


num_thread = 6
l = len(png_files)
def read_thread(PID):
    for i in range(PID, l, num_thread):
        # read
	img = cv2.imread('/scratch/cluster/zharucs/data/' + png_files[i])
	# initialize
	imgsize = img.shape
	img_width  = imgsize[1]
	img_height = imgsize[0]
	sm = pySaliencyMap(img_width, img_height)
	# computation
	saliency_map = sm.SMGetSM(img)
	saliency_map = cv2.resize(saliency_map, RESIZE_SHAPE)
	saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)

	cv2.imwrite(save_path + png_files[i], saliency_map)

	# print the processing bar
       	print "\r%d/%d" % (i,l),
        sys.stdout.flush()

print "Saving bottom up saliency maps for dataset %s" % dataset_name
print "Processing: \t"
t1 = time.time()
o=ForkJoiner(num_thread=num_thread, target=read_thread)
o.join()
print "Done. Saving bottom up saliency maps takes %.1fs" % (time.time()-t1)
