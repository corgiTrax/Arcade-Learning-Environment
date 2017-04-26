#!/usr/bin/env python
import sys, time
import tensorflow as tf, numpy as np
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from IPython import embed
import tensorflow.contrib.layers as L
import input_utils 

NUM_CLASSES = 6
LABELS_FILE_TRAIN = 'mem_dataset/3_Apr-13-18-54-21-train.txt'
LABELS_FILE_VAL = 'mem_dataset/3_Apr-13-18-54-21-val.txt'
MODEL_DIR = 'tflearn_modeldir/3'
SHAPE = (210,160,3) # height * width * channel This cannot read from file and needs to be provided here

class Dataset:
  train_imgs, train_lbl, train_size = None, None, None
  val_imgs, val_lbl, val_size = None, None, None
  def read_to_memory(self):
    print "Reading all training data into memory..."
    self.train_imgs, self.train_lbl = input_utils.read_np(LABELS_FILE_TRAIN)
    self.train_size = len(self.train_lbl)
    print "Reading all validation data into memory..."
    self.val_imgs, self.val_lbl = input_utils.read_np(LABELS_FILE_VAL)
    self.val_size = len(self.val_lbl)


def model_fn(features, targets, mode, params):
  data = features['img'] if isinstance(features, dict) else features

  conv1 = L.conv2d(data, num_outputs=20, kernel_size=8, stride=4, padding="SAME", 
    activation_fn=tf.nn.relu, normalizer_fn=L.batch_norm)
  conv2 = L.conv2d(conv1, num_outputs=40, kernel_size=4, stride=2, padding="SAME", 
    activation_fn=tf.nn.relu, normalizer_fn=L.batch_norm)
  conv3 = L.conv2d(conv2, num_outputs=80, kernel_size=3, stride=2, padding="SAME", 
    activation_fn=tf.nn.relu, normalizer_fn=L.batch_norm)
  flattened = L.flatten(conv3)
  fc1 = L.fully_connected(flattened, num_outputs=256, activation_fn=tf.nn.relu) 
  logits = L.fully_connected(fc1, num_outputs=NUM_CLASSES, activation_fn=None)

  predictions = tf.cast(tf.argmax(logits, axis=1),tf.int32)
  predictions_dict = {"action": predictions}

  loss = tf.losses.sparse_softmax_cross_entropy(targets, logits)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(targets, predictions)
  }

  train_op = L.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=None,
      optimizer=tf.train.GradientDescentOptimizer(0.01))

  return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":

  tf.logging.set_verbosity(tf.logging.INFO)

  d=Dataset()
  d.read_to_memory()
  model_params = {} # becomes the "params" argument in model_fn()
  nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params, model_dir=MODEL_DIR)

  nn.fit(x=d.train_imgs,y=d.train_lbl, batch_size=100, steps=1000)
  nn.evaluate(x=d.val_imgs,y=d.val_lbl, batch_size=100)
  nn.fit(x=d.train_imgs,y=d.train_lbl, batch_size=100, steps=1000)
  nn.evaluate(x=d.val_imgs,y=d.val_lbl, batch_size=100)
  nn.fit(x=d.train_imgs,y=d.train_lbl, batch_size=100, steps=1000)
  nn.evaluate(x=d.val_imgs,y=d.val_lbl, batch_size=100)
  nn.fit(x=d.train_imgs,y=d.train_lbl, batch_size=100, steps=1000)
  nn.evaluate(x=d.val_imgs,y=d.val_lbl, batch_size=100)
  embed()

