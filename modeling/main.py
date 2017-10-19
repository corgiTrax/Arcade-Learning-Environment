import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

# This file runs a "fv2ch" model that produces good but not the best result on our AAAI paper.
# It can be run with gocondor.py <main*.py>, a script submiting condor jobs which call this file
# with different paramenters. This file can also be run independently.

NUM_CLASSES=18
BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/" + sys.argv[1]
fv_BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_action/FVs/" + sys.argv[1]
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt'
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt'
fv_LABELS_FILE_TRAIN = fv_BASE_FILE_NAME + '-train.txt'
fv_LABELS_FILE_VAL =  fv_BASE_FILE_NAME + '-val.txt'
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
SHAPE = (84,84,1) # height * width * channel. Cannot be read from file; Needs to be provided here
BATCH_SIZE=100
num_epoch = 50
MODEL_DIR = sys.argv[2]
dropout = float(sys.argv[3])
save_model = True if '--save' in sys.argv else False # you can specify "--save" in argument

MU.keras_model_serialization_bug_fix()
expr = MU.ExprCreaterAndResumer(MODEL_DIR,postfix="fv2ch_dr%s" % (str(dropout)))

if True: # I just want to indent
    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    x=L.Conv2D(32, (8,8), strides=4, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    x=L.Conv2D(64, (4,4), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    x=L.Conv2D(64, (3,3), strides=1, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    
    fvinputs=L.Input(shape=SHAPE)
    fvx=fvinputs
    fvx=L.Conv2D(32, (8,8), strides=4, padding='same')(fvx)
    fvx=L.BatchNormalization()(fvx)
    fvx=L.Activation('relu')(fvx)
    fvx=L.Dropout(dropout)(fvx)
    
    fvx=L.Conv2D(64, (4,4), strides=2, padding='same')(fvx)
    fvx=L.BatchNormalization()(fvx)
    fvx=L.Activation('relu')(fvx)
    fvx=L.Dropout(dropout)(fvx)

    fvx=L.Conv2D(64, (3,3), strides=1, padding='same')(fvx)
    fvx=L.BatchNormalization()(fvx)
    fvx=L.Activation('relu')(fvx)

    x=L.Average()([x,fvx])
    x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)
    x=L.Dense(512, activation='relu')(x)
    x=L.Dropout(dropout)(x)
    logits=L.Dense(NUM_CLASSES, name="logits")(x)
    prob=L.Activation('softmax', name="prob")(logits)

    model=Model(inputs=[inputs, fvinputs], outputs=[logits, prob])
    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(loss={"prob":None, "logits": MU.loss_func},
                 optimizer=opt,metrics={"logits": MU.acc_})

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=input_utils.BIU.Dataset(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE)
dfv=input_utils.BIU.Dataset(fv_LABELS_FILE_TRAIN, fv_LABELS_FILE_VAL, SHAPE)

model.fit([d.train_imgs,dfv.train_imgs], d.train_lbl, BATCH_SIZE, epochs=num_epoch,
    validation_data=([d.val_imgs,dfv.val_imgs], d.val_lbl, d.val_weight),
    shuffle=True, sample_weight=d.train_weight, verbose=2,
    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.001),
        MU.PrintLrCallback()])


score = model.evaluate([d.val_imgs, dfv.val_imgs], d.val_lbl, BATCH_SIZE, 0, sample_weight=d.val_weight)
expr.printdebug("eval score:" + str(score))

if save_model:
  expr.save_weight_and_training_config_state(model)
