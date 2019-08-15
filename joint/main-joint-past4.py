import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import ipdb
import json
import sys
import time
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("Tensorflow version: %s" % tf.__version__)
print("Keras version: %s" % K.__version__)

print("Usage: ipython main.py [PredictMode?]")
print("Usage Predict Mode: ipython main.py 1 parameters Model.hdf5")
print("Usage Training Mode: ipython main.py 0 parameters")

GAME_NAME = sys.argv[1]
if GAME_NAME == 'seaquest':
    VAL_DATASET = ['235_RZ_9578441_Feb-07-16-28-09']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/seaquest_{185_197_212}train_{235}val"  
    MODEL_DIR = "Test/seaquest4frame"

LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
PREDICT_FILE_VAL = BASE_FILE_NAME.split('/')[-1] + '-image_past4'
BATCH_SIZE = 50
num_epoch = 70

resume_model = False
predict_mode = int(sys.argv[2]) 
dropout = float(sys.argv[3])
#lr = float(sys.argv[3])
#factor = float(sys.argv[4])
heatmap_shape = 84
k = 4
stride = 1
SHAPE = (84,84,k) # height * width * channel This cannot read from file and needs to be provided here

import input_utils as IU, misc_utils as MU
if not predict_mode: # if train
    expr = MU.BMU.ExprCreaterAndResumer(MODEL_DIR, postfix='pKf_dp%.2f_k%ds%d' % (dropout,k,stride))
    expr.redirect_output_to_logfile_if_not_on("eldar-11")

MU.BMU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

if resume_model:
    model = expr.load_weight_and_training_config_and_state()
    expr.printdebug("Checkpoint found. Resuming model at %s" % expr.dir_lasttime)
else:
    ###############################
    # Architecture of the network #
    ###############################

    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    
    conv1=L.Conv2D(32, (8,8), strides=4, padding='valid')
    x = conv1(x)
    print conv1.output_shape
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x) # TODO may not need batch norm?
    x=L.Dropout(dropout)(x)
    
    conv2=L.Conv2D(64, (4,4), strides=2, padding='valid')
    x = conv2(x)
    print conv2.output_shape
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    conv3=L.Conv2D(64, (3,3), strides=1, padding='valid')
    x = conv3(x)
    print conv3.output_shape
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    y=L.Flatten()(x)
    y=L.Dense(512, activation='relu')(y)
    y=L.Dropout(dropout)(y)
    y=L.Dense(IU.V.NUM_ACTION)(y)
    action_output=L.Activation('softmax', name="act_out")(y)
   
    deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid')
    x = deconv1(x)
    print deconv1.output_shape
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)

    deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid')
    x = deconv2(x)
    print deconv2.output_shape
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)         

    deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid')
    x = deconv3(x)
    print deconv3.output_shape

    gaze_output = L.Activation(MU.my_softmax, name="gaze_out")(x)

    model=Model(inputs=inputs, outputs=[gaze_output, action_output])

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #opt=K.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 

    model.compile(optimizer=opt, \
    loss={"act_out": MU.loss_func, "gaze_out": MU.my_kld},\
    loss_weights={"act_out":1., "gaze_out": 1.},\
    metrics={"act_out": MU.acc_, "gaze_out": MU.my_kld})
    #TODO adjust weights

d=IU.DatasetWithHeatmap_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, heatmap_shape, GAZE_POS_ASC_FILE, k, stride)

if not predict_mode: # if train
    expr.dump_src_code_and_model_def(sys.argv[0], model)

    # TODO unclear why need two weights
    model.fit(d.train_imgs, {"act_out": d.train_lbl, "gaze_out": d.train_GHmap}, BATCH_SIZE, epochs=num_epoch,
        validation_data=(d.val_imgs, {"act_out": d.val_lbl, "gaze_out": d.val_GHmap}, {"act_out": d.val_weight, "gaze_out": d.val_weight}),
        shuffle=True, sample_weight={"act_out": d.train_weight, "gaze_out": d.train_weight}, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00001),
            MU.BMU.PrintLrCallback()])
    
    expr.save_weight_and_training_config_state(model)

    score = model.evaluate(d.val_imgs, {"act_out": d.val_lbl, "gaze_out": d.val_GHmap}, BATCH_SIZE, 0, sample_weight={"act_out": d.val_weight, "gaze_out": d.val_weight})
    expr.printdebug("eval score:" + str(score))

elif predict_mode: # if predict
    model.load_weights(sys.argv[4])

    print "Evaluating model..."
    #train_score = model.evaluate(d.train_imgs, d.train_GHmap, BATCH_SIZE, 0)
    val_score = model.evaluate(d.val_imgs, {"act_out": d.train_lbl, "gaze_out": d.train_GHmap}, BATCH_SIZE, 0, d.val_weight)
    #print "Train loss is:  " , train_score
    print "Val loss is: " , val_score

    print "Predicting results..."
    #train_pred = model.predict(d.train_imgs, BATCH_SIZE) # [1] for prob
    val_pred = model.predict(d.val_imgs, BATCH_SIZE)
    print "Predicted."

    print "Converting predicted results into png files and save..."
    #IU.save_heatmap_png_files(d.train_fid, train_pred, TRAIN_DATASET, 'saliency/')
    IU.save_heatmap_png_files(d.val_fid, val_pred, VAL_DATASET, 'saliency/')
    print "Done."

    print "Writing predicted results into the npz file..."
    np.savez_compressed(PREDICT_FILE_VAL, fid=d.val_fid, heatmap=val_pred)
    print "Done. Output is:"
    print " %s" % PREDICT_FILE_VAL+'.npz'
