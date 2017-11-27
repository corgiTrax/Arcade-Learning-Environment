import numpy as np
import input_utils
import keras as K
import copy as cp
from misc_utils import my_kld
import time, sys
import tensorflow as tf


BASE_FILE_NAME = sys.argv[1]
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
PREDICT_FILE_VAL = BASE_FILE_NAME + '-random'
SHAPE = (84,84,1)
heatmap_shape = 84
sess = tf.Session()

d=input_utils.DatasetWithHeatmap(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, heatmap_shape, GAZE_POS_ASC_FILE)

val_predict = np.zeros([d.val_size,heatmap_shape,heatmap_shape,1], dtype=np.float32)
for i in range(d.val_size):
        random_predict = np.random.randint(low=0, high=6, size=(heatmap_shape, heatmap_shape))
	random_predict = random_predict*1.0 / random_predict.sum()
        #random_predict = np.ones([heatmap_shape, heatmap_shape],dtype=np.float32)
        #random_predict = random_predict / 7056
	#print random_predict.sum()
	val_predict[i,:,:,0] = random_predict

t1 = time.time()
print "Writing predicted results into the npz file..."
np.savez_compressed(PREDICT_FILE_VAL, fid=d.val_fid, heatmap=val_predict)
print "Done. Time spent to write npz file: %.1fs" % (time.time()-t1)
print "Outputs are:"
print " %s" % PREDICT_FILE_VAL+'.npz'

val_predict = tf.stack(val_predict)
d.val_GHmap = tf.stack(d.val_GHmap)
val_result = my_kld(d.val_GHmap, val_predict).eval(session = sess)

val_result = np.asarray(val_result, dtype=np.float32)
print "Random saliency predict KL-divergence on val: %f" % val_result.mean()
