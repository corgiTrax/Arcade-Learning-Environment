#!/usr/bin/env python
import sys, time
import tensorflow as tf, numpy as np
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from InputPipeline import *
from IPython import embed
import tensorflow.contrib.layers as L

NUM_CLASSES = 4


def input_fn():
	train_image_batch, train_label_batch, TRAIN_SIZE = create_input_pipeline(LABELS_FILE_TRAIN, batch_size=100, num_epochs=None)
	return {"img": train_image_batch}, train_label_batch

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


	model_params = {"learning_rate": 0.01}
	nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)
	nn.fit(input_fn=input_fn, steps=500)





		