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

print("Usage: ipython main.py gamename [PredictMode?]")
print("Usage Predict Mode: ipython main.py gamename 1 parameters Model.hdf5")
print("Usage Training Mode: ipython main.py gamename 0 parameters")

GAME_NAME = sys.argv[1]
if GAME_NAME == 'seaquestAvg':
    VAL_DATASET = ['87_RZ_3435454_Aug-22-16-01-22']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/seaquest_\{75_83_86\}train_\{87\}val"  
    MODEL_DIR = "ColorAvg_Test/SeaquestAvg"
elif GAME_NAME == 'seaquestNoAvg':
    VAL_DATASET = ['235_RZ_9578441_Feb-07-16-28-09']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/seaquest_\{185_197_212\}train_\{235\}val"  
    MODEL_DIR = "ColorAvg_Test/SeaquestNoAvg"


LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
AFFIX = '-image+opf_past4'
PREDICT_FILE_VAL = BASE_FILE_NAME.split('/')[-1] + AFFIX
BATCH_SIZE = 50
num_epoch = 70

resume_model = False
predict_mode = int(sys.argv[2]) 
dropout = float(sys.argv[3])
heatmap_shape = 84
k = 4
stride = 1
SHAPE = (84,84,k) # height * width * channel This cannot read from file and needs to be provided here

import input_utils as IU, misc_utils as MU
if not predict_mode: # if train
    expr = MU.BMU.ExprCreaterAndResumer(MODEL_DIR, postfix="pKf_dp%.1f_k%ds%d" % (dropout,k,stride))
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
    # First channel: image

    x_inputs=L.Input(shape=SHAPE)    
    x=x_inputs #inputs is used by the line "Model(inputs, ... )" below
    
    conv11=L.Conv2D(32, (8,8), strides=4, padding='valid')
    x = conv11(x)
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    conv12=L.Conv2D(64, (4,4), strides=2, padding='valid')
    x = conv12(x)
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    conv13=L.Conv2D(64, (3,3), strides=1, padding='valid')
    x = conv13(x)
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    deconv11 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid')
    x = deconv11(x)
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)

    deconv12 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid')
    x = deconv12(x)
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)         

    deconv13 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid')
    x = deconv13(x)
    print deconv13.output_shape
    x = L.Activation('relu')(x)
    x_output = L.BatchNormalization()(x)

    # Channel 2: optical flow
    y_inputs=L.Input(shape=SHAPE)
    y=y_inputs # inputs is used by the line "Model(inputs, ... )" below
    
    conv21=L.Conv2D(32, (8,8), strides=4, padding='valid')
    y = conv21(y)
    y=L.Activation('relu')(y)
    y=L.BatchNormalization()(y)
    y=L.Dropout(dropout)(y)
    
    conv22=L.Conv2D(64, (4,4), strides=2, padding='valid')
    y = conv22(y)
    y=L.Activation('relu')(y)
    y=L.BatchNormalization()(y)
    y=L.Dropout(dropout)(y)
    
    conv23=L.Conv2D(64, (3,3), strides=1, padding='valid')
    y = conv23(y)
    y=L.Activation('relu')(y)
    y=L.BatchNormalization()(y)
    y=L.Dropout(dropout)(y)
    
    deconv21 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid')
    y = deconv21(y)
    y=L.Activation('relu')(y)
    y=L.BatchNormalization()(y)
    y=L.Dropout(dropout)(y)

    deconv22 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid')
    y = deconv22(y)
    y=L.Activation('relu')(y)
    y=L.BatchNormalization()(y)
    y=L.Dropout(dropout)(y)         

    deconv23 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid')
    y = deconv23(y)
    print deconv23.output_shape
    y = L.Activation('relu')(y)
    y_output = L.BatchNormalization()(y)

    # Merge outputs from 2 channels
    outputs = L.Average()([x_output, y_output])
    outputs = L.Activation(MU.my_softmax)(outputs)

    model=Model(inputs=[x_inputs,y_inputs], outputs=outputs)

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    # opt=K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        
    model.compile(loss=MU.my_kld, optimizer=opt, metrics=[MU.NSS])
    
d=IU.DatasetWithHeatmap_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, heatmap_shape, GAZE_POS_ASC_FILE, k, stride)
of=IU.Dataset_OpticalFlow_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, k, stride)

if not predict_mode: # if train
    expr.dump_src_code_and_model_def(sys.argv[0], model)

    model.fit([d.train_imgs,of.train_flow], d.train_GHmap, BATCH_SIZE, epochs=num_epoch,
        validation_data=([d.val_imgs,of.val_flow], d.val_GHmap, d.val_weight),
        shuffle=True, sample_weight=d.train_weight, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00001),
            MU.BMU.PrintLrCallback()])

    expr.save_weight_and_training_config_state(model)

    score = model.evaluate([d.val_imgs,of.val_flow], d.val_GHmap, BATCH_SIZE, 0, sample_weight=d.val_weight)
    expr.printdebug("eval score:" + str(score))

elif predict_mode: # if predict
    model.load_weights(sys.argv[4])

    print "Evaluating model..."
    #train_score = model.evaluate([d.train_imgs, of.train_flow], d.train_GHmap, BATCH_SIZE, 0, sample_weight=d.train_weight)
    #print "Train loss is:  " , train_score
    val_score = model.evaluate([d.val_imgs,of.val_flow], d.val_GHmap, BATCH_SIZE, 0, sample_weight=d.val_weight)
    print "Val loss is: " , val_score

    print "Predicting results..."
    train_pred = model.predict([d.train_imgs, of.train_flow], BATCH_SIZE) 
    val_pred = model.predict([d.val_imgs, of.val_flow], BATCH_SIZE)
    print "Predicted."

    # Uncomment this block to save predicted gaze heatmap for visualization
    #print "Converting predicted results into png files and save..."
    #IU.save_heatmap_png_files(d.train_fid, train_pred, TRAIN_DATASET, '../../saliency/')
    #IU.save_heatmap_png_files(d.val_fid, val_pred, VAL_DATASET, '../../saliency/')
    #print "Done."

    print "Writing predicted gaze heatmap (train) into the npz file..."
    np.savez_compressed(BASE_FILE_NAME.split('/')[-1] + '-train' + AFFIX, fid=d.train_fid, heatmap=train_pred)
    print "Done. Output is:"
    print " %s" % BASE_FILE_NAME.split('/')[-1] + '-train' + AFFIX + '.npz'

    print "Writing predicted gaze heatmap (val) into the npz file..."
    np.savez_compressed(BASE_FILE_NAME.split('/')[-1] + '-val' + AFFIX, fid=d.val_fid, heatmap=val_pred)
    print "Done. Output is:"
    print " %s" % BASE_FILE_NAME.split('/')[-1] + '-val' + AFFIX + '.npz'
