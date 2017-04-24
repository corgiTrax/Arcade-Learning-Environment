import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

NUM_CLASSES=5
LABELS_FILE_TRAIN = '/scratch/cluster/zhuode93/dataset/3_Apr-13-18-54-21-train.txt'
LABELS_FILE_VAL = '/scratch/cluster/zhuode93/dataset/3_Apr-13-18-54-21-val.txt'
SHAPE = (210,160,3) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE=100
num_epoch = 40
MODEL_DIR = 'Expr_3_sgd'
resume_model = False

MU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

expr = MU.ExprCreaterAndResumer(MODEL_DIR, postfix="2d_nestF_DQN")
sys.stdout, sys.stderr = expr.logfile, expr.logfile

if resume_model:
    model = expr.load_weight_and_training_config_and_state()
    expr.printdebug("Checkpoint found. Resuming model at %s" % expr.dir_lasttime)
else:
    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    x=L.Conv2D(32, (8,8), strides=4, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Conv2D(64, (4,4), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Conv2D(64, (3,3), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Flatten()(x)
    x=L.Dense(256, activation='relu')(x)
    x=L.Dropout(0.5)(x)
    x=L.Dense(256, activation='relu')(x)
    x=L.Dropout(0.5)(x)
    logits=L.Dense(NUM_CLASSES, name="logits")(x)
    prob=L.Activation('softmax', name="prob")(logits)
    model=Model(inputs=inputs, outputs=[logits, prob])

    model.compile(loss={"prob":None, "logits": MU.loss_func},
                 optimizer=K.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), # nesterov=True converges faster
                 metrics={"logits": MU.acc_})

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=input_utils.Dataset(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE)
model.fit(d.train_imgs, d.train_lbl, batch_size=100, epochs=60,
    validation_data=(d.val_imgs, d.val_lbl),
    shuffle=True,verbose=2,
    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5, min_lr=0.001),
        MU.PrintLrCallback()])

expr.save_weight_and_training_config_state(model)

score = model.evaluate(d.val_imgs, d.val_lbl, BATCH_SIZE, 0)
expr.printdebug("eval score:" + str(score))

# TODO start to test different optimizers, model archetecture, avoid overfitting
# TODO monitor loss at tensorboard