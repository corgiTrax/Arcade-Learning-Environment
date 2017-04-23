import tensorflow as tf, numpy as np, keras as K
from IPython import embed
import ipdb

def acc(y_true, y_pred):
  return tf.reduce_mean(
    tf.cast(tf.nn.in_top_k(
      targets=tf.squeeze(tf.cast(y_true,tf.int32)), 
      predictions=y_pred,k=1),tf.float32))

def save_GPU_mem_keras():
    # don't let tf eat all the memory on eldar-11
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.backend.set_session(sess)
