import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import ipdb
import json
import sys
import misc_utils as MU
import input_utils as IU
import time

print("Usage: ipython main.py [PredictMode?]")
print("Usage Predict Mode: ipython main.py 1 parameters Model.hdf5")
print("Usage Training Mode: ipython main.py 0 parameters")

GAME_NAME = sys.argv[1]
if GAME_NAME == 'seaquest':
    VAL_DATASET = ['75_RZ_3006069_Aug-17-16-46-05']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/seaquest_mixall"  
    MODEL_DIR = "Mixall"

LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
PREDICT_FILE_TRAIN = BASE_FILE_NAME + '-train-image_past4'
PREDICT_FILE_VAL = BASE_FILE_NAME + '-val-image_past4'
BATCH_SIZE = 50
num_epoch = 70

resume_model = False
heatmap_shape = 84
k = 4
stride = 1
SHAPE = (84,84,k) # height * width * channel This cannot read from file and needs to be provided here


MU.BMU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

d=IU.DatasetWithHeatmap_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, heatmap_shape, GAZE_POS_ASC_FILE, k, stride) 
np.save('mean_files/'+GAME_NAME+'.mean', d.mean)

# if there is optical flow
# of=IU.Dataset_OpticalFlow_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, k, stride)
# np.save('mean_files/'+GAME_NAME+'.of.mean', of.mean)

