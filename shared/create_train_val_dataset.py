#!/usr/bin/env python

import sys, re, tarfile, os, shutil, subprocess, threading
from base_input_utils import read_gaze_data_asc_file, frameid_from_filename, rescale_and_clip_gaze_pos
from IPython import embed

def untar(tar_path, output_path):
    tar = tarfile.open(tar_path, 'r')
    tar.extractall(output_path)
    png_files = [png for png in tar.getnames() if png.endswith('.png')]
    png_files = sorted(png_files,key=frameid_from_filename)
    return png_files

def use_spec_file():

    print ("Reading dataset specification file...")
    spec_file, dataset_name, output_path = sys.argv[2], sys.argv[3], sys.argv[4]
    lines = []
    with open(spec_file,'r') as f:
        for line in f:
            if line.strip().startswith("#") or line == "": 
                continue # skip comments or empty lines
            lines.append(line)
    spec = '\n'.join(lines)
    import ast
    spec = ast.literal_eval(spec)
    # some simple sanity checks
    assert isinstance(spec,list)
    for fname in [e['TAR'] for e in spec] + [e['ASC'] for e in spec]:
        if not os.path.exists(fname): 
            raise IOError("No such file: %s" % fname)
    
    print ("Untaring files in parallel...")
    png_files_each=[None]*len(spec)
    def untar_thread(PID):
        png_files_each[PID] = untar(spec[PID]['TAR'], output_path)
    untar_work=ForkJoiner(num_thread=len(spec), target=untar_thread)

    print ("Reading asc files while untaring...")
    frameid2action_each=[None]*len(spec)
    frameid2pos_each=[[]]*len(spec)
    for i in range(len(spec)):
        frameid2pos_each[i], frameid2action_each[i], _, _, _ = read_gaze_data_asc_file(spec[i]['ASC'])

    print ("Concatenating asc file while untaring...")
    asc_filename = output_path+'/'+dataset_name+'.asc'
    with open(asc_filename, 'w') as f:
        subprocess.call(['cat']+[entry['ASC'] for entry in spec], stdout=f)
    print ("Waiting for untaring to finish...")
    untar_work.join()

    print ("Generating train/val label files...")
    xy_str_train = []
    xy_str_val =   []
    BAD_GAZE = (-1,-1)
    RESIZE_SHAPE = (84,84)
    for i in range(len(spec)):
        # prepare xy_str[] --- all (example, label) strings
        xy_str = []
        for png in png_files_each[i]:
            fid = frameid_from_filename(png)
            if fid in frameid2action_each[i] and frameid2action_each[i][fid] != None:
                # TODO(zhuode): Here, Luxin saves weight in label file, and outputs the last gaze/bad_gaze in label file, but it is not the best
                # place to store such infomation, because the label file should not be designed to meet the need of training a specific model.
                # the label file is used by all models, so we should guarantee it only stores info needed by all models.
                # If a model needs such information, it should compute them in input_utils or store them in another file.
                # An bad scenerio is when we have models that doesn't use the last gaze or the weight in label file, then such info in label file
                # can only adds confusion. 
                if fid in frameid2pos_each[i] and frameid2pos_each[i][fid]:
                    weight = 1
                    # loop to find if there is bad gaze; if there is, then set weight to 0
                    for j in range(len(frameid2pos_each[i][fid])):
                        isbad, _, _ = rescale_and_clip_gaze_pos(frameid2pos_each[i][fid][j][0], frameid2pos_each[i][fid][j][1], RESIZE_SHAPE[0], RESIZE_SHAPE[1])
                        if isbad:
                            frameid2pos_each[i][fid] = [BAD_GAZE]
                            weight = 0
                            break

                    l = len(frameid2pos_each[i][fid])
                    xy_str.append('%s %d %f %f %f' % (png, frameid2action_each[i][fid], frameid2pos_each[i][fid][l-1][0], frameid2pos_each[i][fid][l-1][1], weight))
                else:# if no gaze, set gaze to -1 and weight to 0
                    xy_str.append('%s %d %f %f %f' % (png, frameid2action_each[i][fid], BAD_GAZE[0], BAD_GAZE[1], 0))
            else:
                print ("Warning: Cannot find the label for frame ID %s. Skipping this frame." % str(fid))
        
        # assign each xy_str to the train/val part of the dataset
        def assign(range_list, target):
            # sort the ranges using left bound as key (e.g. ["0.5-1", "0-0.2"] becomes ["0-0.2", "0.5-1"])
            # A must, because Dataset_PastKFramesByTime in input_utils.py assert data_is_sorted_by_timestamp()
            range_list=sorted(range_list, key=lambda x: float(x.split('-')[0]))
            for range_ in range_list:
                l, r = range_.split('-')
                l, r = float(l), float(r)
                target.extend(xy_str[int(l*len(xy_str)):int(r*len(xy_str))])
        assign(spec[i]['TRAIN'], xy_str_train)
        assign(spec[i]['VAL'], xy_str_val)

    train_file_name = output_path + "/" + dataset_name + '-train.txt'
    val_file_name =   output_path + "/" + dataset_name + '-val.txt'

    with open(train_file_name, 'w') as f:
        f.write('# ' + '# '.join(lines) + '\n') # echo spec file content
        f.write('\n'.join(xy_str_train))
        f.write('\n')

    with open(val_file_name, 'w') as f:
        f.write('# ' + '# '.join(lines) + '\n') # echo spec file content
        f.write('\n'.join(xy_str_val))
        f.write('\n')

    print ("\nDone. Outputs are:")
    print (" %s" % asc_filename)
    print (" %s (%d examples)" % (train_file_name, len(xy_str_train)))
    print (" %s (%d examples)" % (val_file_name, len(xy_str_val)))
    print ("For convenience, dataset specification is also prepended to train/val text file.")


class ForkJoiner():
    def __init__(self, num_thread, target):
        self.num_thread = num_thread
        self.threads = [threading.Thread(target=target, args=[PID]) for PID in range(num_thread)]
        for t in self.threads: 
            t.start()
    def join(self):
        for t in self.threads: t.join()

if __name__ == '__main__':
    if len(sys.argv)<5:
        print ("Usage: ")
        print ("  %s --spec text_file(see dataset_specification_example.txt) dataset_name(give a name to this dataset)  output_path(e.g. a directory called 'dataset')\n"  % sys.argv[0])
        sys.exit(0)
    use_spec_file()
