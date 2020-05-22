import numpy as np
import sys
import os
from scipy import misc
import copy as cp

DIR = sys.argv[1]
GAME_NAME = sys.argv[2]
FRAME_CUTOFF = int(sys.argv[3])
dirFiles = os.listdir(DIR)
dirFiles.sort(key=lambda f: int(filter(str.isdigit, f)))
imgfprefix = DIR.split("_")[1] + "_" + DIR.split("_")[2] + "_"

TOTAL_FRAMES = 100
PAST_FRAMES = 15
STEP = int(FRAME_CUTOFF / TOTAL_FRAMES)

frame_ct = 0
frame_id = PAST_FRAMES + 1
all_imgs = []
fname = GAME_NAME + ".txt"
label_file = open(fname,'w')
while frame_ct < TOTAL_FRAMES:
	imgs = []
	for i in range(0,PAST_FRAMES+1):
		cur_frame_id = frame_id - i
		imgfname = imgfprefix + str(cur_frame_id) +".png"
		label_file.write(DIR + imgfname + '\n')
		img = misc.imread(DIR+imgfname)
		imgs.append(img)
	frame_id += STEP
	frame_ct += 1
	all_imgs.append(cp.deepcopy(imgs))

label_file.close()
all_imgs = np.asarray(all_imgs)
print("Shape of saved output: ", all_imgs.shape)
np.savez_compressed("human_data_" + GAME_NAME, raw=all_imgs)



