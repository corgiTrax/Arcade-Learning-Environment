import numpy as np
import input_utils
import keras as K
import tensorflow as tf
import copy as cp

BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{36_RZ}tr_{37_RZ}val"
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
SHAPE = (84,84,1)
heatmap_shape = 84
sess = tf.Session()

def my_kld(y_true, y_pred):
    y_true = K.backend.clip(y_true, K.backend.epsilon(), 1)
    y_pred = K.backend.clip(y_pred, K.backend.epsilon(), 1)
    #print(y_true, y_pred)
    return K.backend.sum(y_true * K.backend.log(y_true / y_pred), axis = [1,2,3])

d=input_utils.DatasetWithHeatmap(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, heatmap_shape, GAZE_POS_ASC_FILE)

# train_result = []
# for i in range(d.train_size):
# 	random_predict = np.random.random_integers(low=0, high=5, size=(84,84))
# 	random_predict = random_predict.astype('float32')
# 	random_predict = random_predict / random_predict.sum()
# 	print random_predict.sum()

# 	random_predict = tf.stack(random_predict)
# 	train_result.append(my_kld(d.train_GHmap[i,:,:,0], random_predict).eval(session = sess))

# train_result = np.asarray(train_result, dtype=np.float32)
# print "Random saliency predict KL-divergence on train: %f" % train_result.mean()

val_predict = np.zeros([d.val_size,heatmap_shape,heatmap_shape,1], dtype=np.float32)
for i in range(d.val_size):
	#random_predict = np.random.random_integers(low=0, high=5, size=(heatmap_shape, heatmap_shape))
	#random_predict = random_predict*1.0 / random_predict.sum()
        random_predict = np.ones([heatmap_shape, heatmap_shape],dtype=np.float32)
        random_predict = random_predict / 7056
	#print random_predict.sum()
	val_predict[i,:,:,0] = random_predict

val_predict = tf.stack(val_predict)
d.val_GHmap = tf.stack(d.val_GHmap)
val_result = my_kld(d.val_GHmap, val_predict).eval(session = sess)

val_result = np.asarray(val_result, dtype=np.float32)
print val_result
print "Random saliency predict KL-divergence on val: %f" % val_result.mean()

mean, var = tf.nn.moments(d.val_GHmap, [1,2,3], keep_dims=True)
stddev = tf.sqrt(var)
sal = (d.val_GHmap - mean) / stddev

val_gazepoints = tf.ceil(d.val_GHmap)
val_gazepoints = val_gazepoints / tf.reduce_sum(val_gazepoints, axis=[1,2,3], keep_dims=True)

score = tf.multiply(d.val_GHmap, sal)
#score = tf.multiply(val_gazepoints, sal)
score = tf.contrib.keras.backend.sum(score, axis=[1,2,3])
r = score.eval(session=sess)
r = np.asarray(r)
for i in range(len(r)):
    print r[i]
print tf.contrib.keras.backend.mean(score).eval(session=sess)

