#!/usr/bin/env python

import sys, re, tarfile, os

def frameid_from_filename(fname): 
    """ Extract '23' from '0_blahblah/23.png' """

    a, b = os.path.splitext(os.path.basename(fname))
    try:
        frameid = int(a)
    except ValueError as ex:
        raise ValueError("cannot convert filename '%s' to frame ID (an integer)" % fname)
    return frameid

def read_gaze_data_asc_file(fname):
    """ This function reads a ASC file and returns a dictionary mapping frame ID to gaze position """

    with open(fname, 'r') as f:
        lines = f.readlines()
    frameid = "BEFORE-FIRST-FRAME"
    frameid2action = {frameid: None}

    for (i,line) in enumerate(lines):

        match_scr_msg = re.match("MSG\s+(\d+)\s+SCR_RECORDER FRAMEID (\d+)", line)
        if match_scr_msg: # when a new id is encountered
            timestamp, frameid = match_scr_msg.group(1), match_scr_msg.group(2)
            frameid = int(frameid)
            frameid2action[frameid] = None
            continue

        match_action = re.match("MSG\s+(\d+)\s+key_pressed atari_action (\d+)", line)
        if match_action:
            timestamp, action_label = match_action.group(1), match_action.group(2)
            if frameid2action[frameid] is None:
            	frameid2action[frameid] = int(action_label)
            else:
            	print "Warning: there are more than 1 action for frame id %d. Not supposed to happen." % frameid
            continue

    return frameid2action

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
	frameid2action = read_gaze_data_asc_file(asc_file)

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
