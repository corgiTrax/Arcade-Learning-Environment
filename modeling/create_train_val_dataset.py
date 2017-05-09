#!/usr/bin/env python

import sys, re, tarfile, os
from input_utils import read_gaze_data_asc_file, frameid_from_filename

def untar(tar_path, output_path):
    tar = tarfile.open(tar_path, 'r')
    tar.extractall(output_path)
    png_files = [png for png in tar.getnames() if png.endswith('.png')]
    return png_files

if __name__ == '__main__':
	if len(sys.argv)<5: 
		print "Usage: %s asc_file tar_file output_path(e.g. a directory called 'dataset') training_data_percentage(float, range [0.0, 1.0])" % sys.argv[0]
		sys.exit(0)
	asc_file, tar_file, output_path, percent = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
	
	print "reading asc_file..."
	_, frameid2action = read_gaze_data_asc_file(asc_file)

	print "Untaring file..."
	png_files = untar(tar_file, output_path)

	print "Generating train/val label files..."
	xy_str = []
	for png in png_files:
		fid = frameid_from_filename(png)
		if fid in frameid2action and frameid2action[fid] != None:
			xy_str.append('%s %d' % (png, frameid2action[fid]))
		else:
			print "Warning: Cannot find the label for frame ID %d. Skipping this frame." % fid
	
	xy_str_train = xy_str[:int(percent*len(xy_str))]
	xy_str_val =   xy_str[int(percent*len(xy_str)):]
	asc_filename, _ = os.path.splitext(os.path.basename(asc_file))
	train_file_name = output_path + "/" + asc_filename + '-train.txt'
	val_file_name =   output_path + "/" + asc_filename + '-val.txt'

	with open(train_file_name, 'w') as f:
			f.write('\n'.join(xy_str_train))

	with open(val_file_name, 'w') as f:
			f.write('\n'.join(xy_str_val))

	print "Done. Outputs are:\n %s (%d examples)\n %s (%d examples)" % (train_file_name, len(xy_str_train), val_file_name, len(xy_str_val))
