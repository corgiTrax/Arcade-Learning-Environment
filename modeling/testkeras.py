from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D
import numpy as np


batch_size = 100
num_classes = 10
epochs = 2

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# # Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# model = Sequential()
# model.add(Flatten(input_shape=x_train.shape[1:]))
# model.add(Dense(512))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(Dense(512))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(Dense(num_classes, activation="softmax", name='prob'))

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

# Let's train the model using RMSprop
model.compile(loss=[None, 'sparse_categorical_crossentropy'],
              optimizer=sgd,
              metrics={"prob":'accuracy'})

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
mean, std = np.mean(x_train, axis=(0,1,2)), np.std(x_train, axis=(0,1,2))
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)