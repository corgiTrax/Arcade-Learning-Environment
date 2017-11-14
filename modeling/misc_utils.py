import tensorflow as tf, numpy as np, keras as K
import shutil, os, time, re, sys
from IPython import embed
import ipdb
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
import base_misc_utils as BMU
 
def keras_model_serialization_bug_fix(): # stupid keras
    # we need to call these functions so that a model can be correctly saved and loaded
    from keras.utils.generic_utils import get_custom_objects
    f=lambda obj_to_serialize: \
        get_custom_objects().update({obj_to_serialize.__name__: obj_to_serialize})
    f(loss_func); f(acc_); f(top2acc_)
    f(loss_func_nonsparse)
    f(acc_nonsparse_wrong)

def loss_func(target, pred): 
    return K.backend.sparse_categorical_crossentropy(output=pred, target=target, from_logits=True)

# This is function is used in ale/modeling/pyModel/main-SmoothLabel.py, because in that case
# the target label is a prob distribution rather than a number
def loss_func_nonsparse(target, pred): 
    return K.backend.categorical_crossentropy(output=pred, target=target, from_logits=True)

def acc_(y_true, y_pred): # don't rename it to acc or accuracy (otherwise stupid keras will replace this func with its own accuracy function when serializing )
  return tf.reduce_mean(
    tf.cast(tf.nn.in_top_k(
      targets=tf.squeeze(tf.cast(y_true,tf.int32)), 
      predictions=y_pred,k=1),tf.float32))

# This is function is used in ale/modeling/pyModel/main-SmoothLabel.py, because in that case
# the target label is a prob distribution rather than a number, so there is no "accuracy" defined.
# and I just want to implement a wrong but approx accuracy here, by pretending the argmax() of y_true
# is the true label. 
def acc_nonsparse_wrong(y_true, y_pred):  
  return tf.reduce_mean(
    tf.cast(tf.nn.in_top_k(
      targets=tf.squeeze(tf.cast(tf.argmax(y_true, axis=1),tf.int32)), 
      predictions=y_pred,k=1),tf.float32))

def top2acc_(y_true, y_pred):
  return tf.reduce_mean(
    tf.cast(tf.nn.in_top_k(
      targets=tf.squeeze(tf.cast(y_true,tf.int32)),
      predictions=y_pred,k=2),tf.float32))
