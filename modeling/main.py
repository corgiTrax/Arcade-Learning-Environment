import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

NUM_CLASSES=5
LABELS_FILE_TRAIN = 'mem_dataset/3_Apr-13-18-54-21-train.txt'
LABELS_FILE_VAL = 'mem_dataset/3_Apr-13-18-54-21-val.txt'
SHAPE = (210,160,3) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE=100
MODEL_DIR = 'expr_3_sgd'
resume_model = True

MU.save_GPU_mem_keras()

def lossfunc(target, pred): 
    return K.backend.sparse_categorical_crossentropy(output=pred,target=target, from_logits=True)
MU.serialize_model_keras_bug_fix(lossfunc)

expr = MU.ExprCreaterAndResumer(MODEL_DIR)
if resume_model:
    model = expr.load_weight_and_training_config_and_state()
    expr.printdebug("Checkpoint found. Resuming model at %s" % expr.dir_lasttime)
else:
    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    x=L.Conv2D(20, (8,8), strides=4, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Conv2D(40, (4,4), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Conv2D(80, (3,3), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Flatten()(x)
    x=L.Dense(256, activation='relu')(x)
    logits=L.Dense(NUM_CLASSES, name="logits")(x)
    prob=L.Activation('softmax', name="prob")(logits)
    model=Model(inputs=inputs, outputs=[logits, prob])

    sgd = K.optimizers.SGD(lr=0.01)
    model.compile(loss={"prob":None, "logits": lossfunc },
                 optimizer=sgd,
                 metrics={"logits": MU.acc})

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=input_utils.Dataset(LABELS_FILE_TRAIN, LABELS_FILE_VAL)
model.fit(d.train_imgs, d.train_lbl, batch_size=100, epochs=10,
    validation_data=(d.val_imgs, d.val_lbl),
    shuffle=True)

expr.save_weight_and_training_config_state(model)

score = model.evaluate(d.val_imgs, d.val_lbl, BATCH_SIZE)
expr.printdebug("eval score:" + str(score))

