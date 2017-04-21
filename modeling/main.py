#!/usr/bin/env python
import sys, time
import tensorflow as tf, numpy as np
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from InputPipeline import *
from IPython import embed
import tensorflow.contrib.layers as L
NUM_CLASSES = 6
LABELS_FILE_TRAIN = 'mem_dataset/6_Apr-13-19-14-59-train.txt'
LABELS_FILE_VAL = 'mem_dataset/6_Apr-13-19-14-59-val.txt'
MODEL_DIR = 'modeldir/6'
SHAPE = (210,160,3) # height * width * channel This cannot read from file and needs to be provided here

class Dataset:
  train_image_batch = None
  train_label_batch = None
  train_size = None
  val_image_batch = None
  val_label_batch = None
  val_size = None
  def train_input_fn(self, recreate=False, num_epochs=None):
    if self.train_image_batch is None or recreate: # cache the result, otherwise tf.contrib.learn.fit will call input_fn() every time
      printdebug("create_input_pipeline() for training")
      self.train_image_batch, self.train_label_batch, self.train_size = create_input_pipeline(LABELS_FILE_TRAIN, SHAPE, batch_size=100, num_epochs=num_epochs)
    return {"img": self.train_image_batch}, self.train_label_batch
  def val_input_fn(self, recreate=False, num_epochs=1):
    if self.val_image_batch is None or recreate:
      printdebug("create_input_pipeline() for validation")
      self.val_image_batch, self.val_label_batch, self.val_size = create_input_pipeline(LABELS_FILE_VAL, SHAPE, batch_size=100, num_epochs=num_epochs)
    return {"img": self.val_image_batch}, self.val_label_batch



def model_fn(features, targets, mode, params):
  """Model function for Estimator."""

  data = features['img']
  conv1 = L.conv2d(data, num_outputs=20, kernel_size=8, stride=4, padding="SAME", 
    activation_fn=tf.nn.relu, normalizer_fn=L.batch_norm)
  conv2 = L.conv2d(conv1, num_outputs=40, kernel_size=4, stride=2, padding="SAME", 
    activation_fn=tf.nn.relu, normalizer_fn=L.batch_norm)
  conv3 = L.conv2d(conv2, num_outputs=80, kernel_size=3, stride=2, padding="SAME", 
    activation_fn=tf.nn.relu, normalizer_fn=L.batch_norm)


  flattened = L.flatten(conv3)
  fc1 = L.fully_connected(flattened, num_outputs=256, activation_fn=tf.nn.relu) 
  logits = L.fully_connected(fc1, num_outputs=NUM_CLASSES, activation_fn=None)


  predictions = tf.argmax(logits, axis=1)
  predictions_dict = {"action": predictions}

  # Calculate loss using mean squared error
  loss = tf.losses.sparse_softmax_cross_entropy(targets, logits)

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(targets, predictions)
  }

  train_op = L.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

  return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
  if len(sys.argv) < 1:
    print "Usage: "
    sys.exit(0)

  tf.logging.set_verbosity(tf.logging.INFO)
  setlogfile('printdebug.txt')

  d=Dataset()
  model_params = {"learning_rate": 0.01}
  nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params, model_dir=MODEL_DIR)
  #nn.fit(input_fn=d.train_input_fn, steps=10000)
  nn.evaluate(input_fn=lambda:d.train_input_fn(recreate=True, num_epochs=1))
  nn.evaluate(input_fn=lambda:d.val_input_fn(recreate=True))

