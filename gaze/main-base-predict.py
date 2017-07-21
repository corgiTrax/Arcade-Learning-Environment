import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

BASE_FILE_NAME = "/Users/zhangluxin/Desktop/ale/dataset_gaze/cat{42_RZ}tr_{44_RZ}val"
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
PREDICT_FILE_TRAIN = BASE_FILE_NAME + '-train-result.txt'
PREDICT_FILE_VAL = BASE_FILE_NAME + '-val-result.txt'
# GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
SHAPE = (84,84,1) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE=100
num_epoch = 70
dropout = 0.45 # turn off dropout for testing
MODEL_DIR = 'Breakout_42_44'
resume_model = False

MU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

# expr = MU.ExprCreaterAndResumer(MODEL_DIR,postfix="baseline")
# expr.redirect_output_to_logfile_if_not_on("eldar-11")

if resume_model:
    model = expr.load_weight_and_training_config_and_state()
    expr.printdebug("Checkpoint found. Resuming model at %s" % expr.dir_lasttime)
else:
    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    x=L.Conv2D(20, (8,8), strides=4, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    x=L.Conv2D(40, (4,4), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    x=L.Conv2D(80, (3,3), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)
    
    x=L.Dense(256, activation='relu')(x)
    x=L.Dropout(dropout)(x)
    logits=L.Dense(2, name="logits")(x)

    model=Model(inputs=inputs, outputs=logits)

    # opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, decay=0.0)
    opt=K.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

    model.load_weights(MODEL_DIR + '/7_baseline/model.hdf5')

# expr.dump_src_code_and_model_def(sys.argv[0], model)

d=input_utils.Dataset(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE)
print "Evaluating model..."
train_score = model.evaluate(d.train_imgs, d.train_gaze, BATCH_SIZE, 0)
val_score = model.evaluate(d.val_imgs, d.val_gaze, BATCH_SIZE, 0)
print "Train loss is: %f " % train_score[0]
print "Val loss is: %f " % val_score[0]

print "Predicting results..."
train_pred = model.predict(d.train_imgs, BATCH_SIZE)
val_pred = model.predict(d.val_imgs, BATCH_SIZE)

xy_str_train = []
for i in range(d.train_size):
    xy_str_train.append('(%d,%d) %f %f' % (d.train_fid[i][0], d.train_fid[i][1], train_pred[i][0], train_pred[i][1]))
xy_str_val = []
for i in range(d.val_size):
    xy_str_val.append('(%d,%d) %f %f' % (d.val_fid[i][0], d.val_fid[i][1], val_pred[i][0], val_pred[i][1]))

print "Writing predicted results into the file..."
with open(PREDICT_FILE_TRAIN, 'w') as f:
    f.write('\n'.join(xy_str_train))
    f.write('\n')
with open(PREDICT_FILE_VAL, 'w') as f:
    f.write('\n'.join(xy_str_val))
    f.write('\n')
print "Done. Outputs are:"
print " %s" % PREDICT_FILE_TRAIN
print " %s" % PREDICT_FILE_VAL

# expr.save_weight_and_training_config_state(model)

# expr.printdebug("eval score:" + str(score))
