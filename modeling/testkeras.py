from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np

from IPython import embed


batch_size = 100
num_classes = 10
epochs = 10

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

inputs=Input(shape=x_train.shape[1:])
x=inputs # inputs is used by the line "Model(inputs, ... )" below
x=Flatten()(x)
x=Dense(512, activation='relu')(x)
x=BatchNormalization()(x)
x=Dense(512, activation='relu')(x)
x=BatchNormalization()(x)
logits=Dense(num_classes, name="logits")(x)
prob=Activation('softmax', name="prob")(logits)
model=Model(inputs=inputs, outputs=[logits, prob])

sgd = keras.optimizers.SGD(lr=0.01)

def acc(y_true, y_pred):
  return tf.reduce_mean(
    tf.cast(tf.nn.in_top_k(
      targets=tf.squeeze(tf.cast(y_true,tf.int32)), 
      predictions=y_pred,k=1),tf.float32))

model.compile(loss={"logits":lambda target, pred: keras.backend.sparse_categorical_crossentropy(output=pred,target=target, from_logits=True)},
          optimizer=sgd,
          metrics={"logits":acc})

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
mean, std = np.mean(x_train, axis=(0,1,2)), np.std(x_train, axis=(0,1,2))
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std

model.fit(x_train, {"logits": y_train},
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test,  {"logits": y_test}),
          shuffle=True)
embed()