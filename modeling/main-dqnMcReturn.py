import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

""" This file is used for training QFunc models suitable for initializing DQN. So it
    uses the same preprocessing as DQN, the same loss function in OpenAI repo (Huber loss)
"""

print("sys.argv is: %s" % str(sys.argv))
NUM_CLASSES=18
BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/seaquest_all"
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt'
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt'
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
SHAPE = (84,84,4) # height * width * channel. Cannot be read from file; Needs to be provided here
BATCH_SIZE=100
num_epoch = 50
MODEL_DIR = "expr_MCreturn"
save_model = True #if '--save' in sys.argv else False # you can specify "--save" in argument

MU.BMU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

expr = MU.BMU.ExprCreaterAndResumer(MODEL_DIR,postfix="BNrelu_penalizeNonSelected")

if True: # I just want to indent
    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    x=L.Conv2D(32, (8,8), strides=4, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Conv2D(64, (4,4), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Conv2D(64, (3,3), strides=1, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Flatten()(x)
    x=L.Dense(512)(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    logits=L.Dense(NUM_CLASSES, name="logits")(x)
    model=Model(inputs=inputs, outputs=[logits])
    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=opt,
        loss={"logits": MU.multi_head_huber_loss_func_with_penalizing_non_selected_Qval}, # DQN training in OpenAI repo uses Huber loss,
        metrics={"logits": MU.maxQval_action_acc}) # so we use the same here for initializing DQN

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=input_utils.DatasetDQN_withMonteCarloReturn(LABELS_FILE_TRAIN, LABELS_FILE_VAL, GAZE_POS_ASC_FILE, discount_factor=0.9)
train_target = np.stack([d.train_lbl, d.train_mc_return],1) # its shape is, if 10000 example,  (10000, 2),
val_target = np.stack([d.val_lbl, d.val_mc_return],1) # which are 10000 tuples of (action_label, mc_return)

model.fit(d.train_imgs, train_target, BATCH_SIZE, epochs=num_epoch,
    validation_data=(d.val_imgs, val_target, d.val_weight),
    shuffle=True, sample_weight=d.train_weight, verbose=2,
    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.001),
        MU.BMU.PrintLrCallback()])


score = model.evaluate(d.val_imgs, val_target, BATCH_SIZE, 0, sample_weight=d.val_weight)
expr.printdebug("eval score:" + str(score))

if save_model:
  expr.save_weight_and_training_config_state(model)
