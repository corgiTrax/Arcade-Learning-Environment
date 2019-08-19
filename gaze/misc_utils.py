import tensorflow as tf, numpy as np, keras as K
import shutil, os, time, re, sys
from IPython import embed
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
import base_misc_utils as BMU

def keras_model_serialization_bug_fix(): # stupid keras
    from keras.utils.generic_utils import get_custom_objects
    f=lambda obj_to_serialize: \
        get_custom_objects().update({obj_to_serialize.__name__: obj_to_serialize})
    f(loss_func)
    f(acc_)
    f(my_softmax)
    f(my_kld)
    f(NSS)

def loss_func(target, pred): 
    return K.backend.sparse_categorical_crossentropy(output=pred, target=target, from_logits=True)

def acc_(y_true, y_pred): # don't rename it to acc or accuracy (otherwise stupid keras will replace this func with its own accuracy function when serializing )
  return tf.reduce_mean(
    tf.cast(tf.nn.in_top_k(
      targets=tf.squeeze(tf.cast(y_true,tf.int32)), 
      predictions=y_pred,k=1),tf.float16))

def my_softmax(x):
    """Softmax activation function. Normalize the whole metrics.
    # Arguments
        x : Tensor.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
   
    return K.activations.softmax(x, axis=[1,2,3])

def my_kld(y_true, y_pred):
    """
    Correct keras bug. Compute the KL-divergence between two metrics.
    """
    epsilon = 1e-10 # introduce epsilon to avoid log and division by zero error
    y_true = K.backend.clip(y_true, epsilon, 1)
    y_pred = K.backend.clip(y_pred, epsilon, 1)
    return K.backend.sum(y_true * K.backend.log(y_true / y_pred), axis = [1,2,3])
        

def NSS(y_true, y_pred):
    """
    This function is to calculate the NSS score of the predict saliency map.

    Input: y_true: ground truth saliency map
           y_pred: predicted saliency map

    Output: NSS score. float num
    """
    
    stddev = tf.contrib.keras.backend.std(y_pred, axis = [1,2,3])
    stddev = tf.expand_dims(stddev, 1)
    stddev = tf.expand_dims(stddev, 2)
    stddev = tf.expand_dims(stddev, 3)
    mean = tf.reduce_mean(y_pred, axis = [1,2,3], keep_dims=True)
    sal = (y_pred - mean) / stddev
    score = tf.multiply(y_true, sal)
    score = tf.contrib.keras.backend.sum(score, axis = [1,2,3])
    
    return score
