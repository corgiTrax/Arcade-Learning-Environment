import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import ipdb
import json
import sys
import misc_utils as MU
import scipy.stats

def my_kld(y_true, y_pred):
    y_true = K.backend.clip(y_true, K.backend.epsilon(), 1)
    y_pred = K.backend.clip(y_pred, K.backend.epsilon(), 1)
    #print(y_true, y_pred)
    return K.backend.sum(y_true * K.backend.log(y_true / y_pred), axis = -1)


A = [[[[0.25],[0.25]],[[0.25],[0.25]]], [[[0.25],[0.25]],[[0.25],[0.25]]], [[[0.25],[0.25]],[[0.25],[0.25]]]]
B = [[[[0.3],[0.3]],[[0.3],[0.1]]], [[[0.25],[0.25]],[[0.25],[0.25]]], [[[0.25],[0.25]],[[0.25],[0.25]]]]

A = np.asarray(A, dtype = np.float32)
B = np.asarray(B, dtype = np.float32)
print(A.shape)
#print(scipy.stats.entropy(A,B))

A = tf.stack(A)
B = tf.stack(B)

sess = tf.Session()

print(my_kld(A,B).eval(session = sess))

