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

def loss_func(target, pred): # should be called cross_entropy_loss_func, but it's too late to change it
    return K.backend.sparse_categorical_crossentropy(output=pred, target=target, from_logits=True)


def multi_head_huber_loss_func(target, pred):
  """ This func assumes about the argument 'target':
      For example, if 10000 example, target's shape is (10000, 2),
      which is 10000 tuples of (action_label, mc_return)
  """
  num_actions = 18 # OK to make this assumption; because when it's not 18, tf will raise errors to let us know
  q_target = target[:,1]
  action_target = tf.cast(target[:,0], tf.uint8)
  q_pred = tf.reduce_sum(pred * tf.one_hot(action_target, num_actions), 1)
  def huber_loss(x, delta=1.0):
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )
  return tf.reduce_mean(huber_loss(q_target - q_pred))

def multi_head_huber_loss_func_with_penalizing_non_selected_Qval(target, pred):
  """ same as multi_head_huber_loss_func() except that it assumes the non_selected_Qval target is zero """
  num_actions = 18
  action_target = tf.cast(target[:,0], tf.uint8)
  q_target = tf.one_hot(action_target, num_actions) * tf.expand_dims(target[:,1],-1)
  def huber_loss(x, delta=1.0):
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )
  return tf.reduce_mean(huber_loss(q_target - pred))


# This is function is used in ale/modeling/pyModel/main-SmoothLabel.py, because in that case
# the target label is a prob distribution rather than a number
def loss_func_nonsparse(target, pred): 
    return K.backend.categorical_crossentropy(output=pred, target=target, from_logits=True)

def acc_(y_true, y_pred): # don't rename it to acc or accuracy (otherwise stupid keras will replace this func with its own accuracy function when serializing )
  return tf.reduce_mean(
    tf.cast(tf.nn.in_top_k(
      targets=tf.squeeze(tf.cast(y_true,tf.int32)), 
      predictions=y_pred,k=1),tf.float32))

def maxQval_action_acc(target, pred):
  """ This function assumes the same as the comment in multi_head_huber_loss_func() 
      The accuracy compted from assuming the action having the maximum Q-value is selected
  """
  return acc_(tf.cast(target[:,0], tf.int32), pred)

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
