import numpy as np
from input_utils import read_heatmap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, os, tarfile
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import cv2

def make_targz_one_by_one(output_filename, source_dir): 
    tar = tarfile.open(output_filename,"w:gz")
    for root,dir,files in os.walk(source_dir):
        for file in files:
            pathfile = os.path.join(root, file)
            tar.add(pathfile)
    tar.close()


def save_heatmap_png_files(heatmap_path, save_path):
	_, person, number, date = save_path.split('_')
	UTID = person + '_' + number
	UTIDhash = hash(UTID)

	print "Reading heatmaps into memory..."
	t1 = time.time()
	heatmaps = read_heatmap(heatmap_path) # heatmaps = {fid: [heatmap]}
	print "Done. Time spent to read heatmaps: %.1fs" % (time.time()-t1)

	if not os.path.exists(save_path):
		os.mkdir(save_path)

	m = cm.ScalarMappable(cmap='hot')

	print "Convolving heatmaps and saving into png files..."
	t2 = time.time()
	for fid in heatmaps:
	 	if fid == 'BEFORE-FIRST-FRAME':
	 		continue

	 	if fid[0] == UTIDhash:
	 		pic = convolve(heatmaps[fid], Gaussian2DKernel(stddev=1))
			pic = m.to_rgba(pic)[:,:,:3]
	 		plt.imsave(save_path+'/' + person+'_'+number+'_' + str(fid[1]) + '.png', pic)
	print "Done. Time spent to save heatmaps: %.1fs" % (time.time()-t2)

	# print "Tar the png files..."
	# t3 = time.time()
	# make_targz_one_by_one(save_path + '.tar.bz2', save_path)
	# print "Done. Time spent to tar files: %.1fs" % (time.time()-t3)


