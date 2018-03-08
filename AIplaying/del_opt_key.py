import h5py
import sys
from os import listdir
from os.path import isfile, join
'''This file takes a directory and read in all .hdf5 files, then delete their 'optimizer_weights' field.
 This fixes a bug when Keras calls a trained model and optimizer_weights dimensions do not match'''

if len(sys.argv) < 2:
 	print("Usage: %s directory_name" % sys.argv[0]) 

mypath = sys.argv[1]

all_files = listdir(mypath)

hdf5_files = []
for f in all_files:
	if isfile(join(mypath, f)) and f.endswith('.hdf5'):
		hdf5_files.append(join(mypath, f))

for f in hdf5_files:
	print("Processing file %s " % f)
	hdf5_file = h5py.File(f, 'r+')
	print("Before operation, this file contains keys: "),
	for key in hdf5_file:
		print(key),
	print
	if 'optimizer_weights' in hdf5_file:
		del hdf5_file['optimizer_weights']
	print("After operation, this file contains keys: "),
	for key in hdf5_file:
		print(key),

	hdf5_file.close()
	print
	print('--------')
