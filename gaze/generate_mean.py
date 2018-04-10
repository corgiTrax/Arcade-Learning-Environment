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
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/seaquest_all"  
    MODEL_DIR = "All_Subject/Seaquest"
elif GAME_NAME == 'mspacman':
    VAL_DATASET = ['76_RZ_3010789_Aug-17-18-01-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/mspacman_all"  
    MODEL_DIR = "All_Subject/Mspacman"
elif GAME_NAME == 'centipede':
    VAL_DATASET = ['80_RZ_3084132_Aug-18-14-23-21']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/centipede_all"  
    MODEL_DIR = "All_Subject/Centipede"
elif GAME_NAME == 'freeway':
    VAL_DATASET = ['72_RZ_2903977_Aug-16-12-25-04']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/freeway_all"  
    MODEL_DIR = "All_Subject/Freeway"
elif GAME_NAME == 'venture':
    VAL_DATASET = ['101_RZ_3603032_Aug-24-14-31-37']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/venture_all"  
    MODEL_DIR = "All_Subject/Venture"    
elif GAME_NAME == 'riverraid':
    VAL_DATASET = ['99_RZ_3590056_Aug-24-10-56-50']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/riverraid_all"  
    MODEL_DIR = "All_Subject/Riverraid"
elif GAME_NAME == 'enduro':
    VAL_DATASET = ['103_RZ_3608911_Aug-24-16-17-04']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/enduro_all"  
    MODEL_DIR = "All_Subject/Enduro"
elif GAME_NAME == 'breakout':
    VAL_DATASET = ['92_RZ_3504740_Aug-23-11-27-56']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/breakout_all"  
    MODEL_DIR = "All_Subject/Breakout" 	

LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
PREDICT_FILE_TRAIN = BASE_FILE_NAME + '-train-image+opf_past4'
PREDICT_FILE_VAL = BASE_FILE_NAME + '-val-image+opf_past4'
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
of=IU.Dataset_OpticalFlow_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, k, stride)
np.save('mean_files/'+GAME_NAME+'.mean', d.mean)
np.save('mean_files/'+GAME_NAME+'.of.mean', of.mean)

