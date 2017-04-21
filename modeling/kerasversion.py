import tensorflow as tf, numpy as np, keras as K
from InputPipeline import *
import keras.layers as L
from keras.models import Model # keras/engine/training.py
from IPython import embed

# don't let tf eat all the memory on eldar-11
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.backend.set_session(sess)

NUM_CLASSES=10
LABELS_FILE_TRAIN = 'mem_dataset/3_Apr-13-18-54-21-train.txt'
LABELS_FILE_VAL = 'mem_dataset/3_Apr-13-18-54-21-val.txt'
SHAPE = (210,160,3) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE=100


# TODO: Eventually I should abandon the whole tensorflow "Input Pipeline" thing and use my own code to create input. Because
# every time calling create_input_pipeline() requires us to run the following code again:
# "sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()]); threads = tf.train.start_queue_runners(sess=sess,coord=coord)"
# but no one wants to do that, which makes the whole "Input Pipeline" thing in tf much more useless; makes "num_epochs" much more useless.
class Dataset:
  train_image_batch = None
  train_label_batch = None
  train_size = None
  val_image_batch = None
  val_label_batch = None
  val_size = None
  def __init__(self):
    self.train_image_batch, self.train_label_batch, self.train_size = create_input_pipeline(LABELS_FILE_TRAIN, SHAPE, BATCH_SIZE, num_epochs=None)
    self.val_image_batch, self.val_label_batch, self.val_size = create_input_pipeline(LABELS_FILE_VAL, SHAPE, BATCH_SIZE, num_epochs=None)
    self.sess = K.backend.get_session()

    # Required
    self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)

  def train_generator(self):
    while True:
        img_np, lbl_np = self.sess.run([self.train_image_batch, self.train_label_batch])
        yield (img_np, lbl_np)

  def val_generator(self):
    # To get strictly accurate result, this function will refuse to generate more when validation data is exhausted
    #for i in range(self.get_legal_val_step()): 
    while True:
        img_np, lbl_np = self.sess.run([self.val_image_batch, self.val_label_batch])
        yield (img_np, lbl_np)

  def get_legal_val_step(self):
    return np.ceil(float(self.val_size) / BATCH_SIZE)


d = Dataset()
inputs = L.Input(SHAPE)
x=L.Conv2D(32, (3, 3), activation='relu', input_shape=SHAPE)(inputs)
x=L.Flatten()(x)
x=L.Dense(64, activation='relu')(x)
x=L.Dropout(0.5)(x)
x=L.Dense(256, activation='relu')(x)
x=L.Dropout(0.5)(x)
logits=(L.Dense(NUM_CLASSES, activation=None))(x)
model=Model(inputs, logits)

sgd = K.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)

model.compile(loss=lambda target, pred: K.backend.sparse_categorical_crossentropy(output=pred,target=target, from_logits=True),
              optimizer=sgd,
              metrics=['accuracy'])
model.fit_generator(d.train_generator(), steps_per_epoch=10, epochs=1)
score = model.evaluate_generator(d.val_generator(), steps=d.get_legal_val_step())
print score # why is this so high?
embed()