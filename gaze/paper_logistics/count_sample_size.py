import sys
import tarfile
import ast
import os

if len(sys.argv) < 3:
    print "Usage: %s label_file_txt spec_file_txt(eg: seaquest_dataset.txt)" % sys.argv[0]
    sys.exit(0)

label_file = sys.argv[1]
spec_file = sys.argv[2]
lines = []
with open(spec_file,'r') as f:
    for line in f:
        if line.strip().startswith("#") or line == "": 
            continue # skip comments or empty lines
        lines.append(line)
spec = '\n'.join(lines)
spec = ast.literal_eval(spec)
# some simple sanity checks
assert isinstance(spec,list)
for fname in [e['TAR'] for e in spec] + [e['ASC'] for e in spec]:
    if not os.path.exists(fname): 
        raise IOError("No such file: %s" % fname)

if label_file.split('-')[-1] == 'train.txt':
    sstr = 'TRAIN'
else:
    sstr = 'VAL'

count_tar = 0
for e in spec:
    if e[sstr][0] == '0-1.0':
        tar = tarfile.open(e['TAR'],'r')
        png_files = tar.getnames()
        print png_files[0]
        count_tar += len(png_files) - 2

count_label = 0
bad_frame = 0
with open(label_file,'r') as f:
    for line in f:
        line=line.strip()
        if line.startswith('#') or line == "":
            continue # skip comments or empty line

        fname, lbl, x, y, w = line.split(' ')
        count_label += 1
        if float(w) == 0.0:
            bad_frame += 1

print "Total image number: %d" % count_tar
print "Total label number: %d" % count_label
print "Bad frame number: %d" % bad_frame
print "Usable data number: %d" % (count_tar - bad_frame)
