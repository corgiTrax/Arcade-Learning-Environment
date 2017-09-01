import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import ipdb
import json
import sys
import time

print("Usage: ipython main.py [PredictMode?]")
print("Usage Predict Mode: ipython main.py 1 parameters Model.hdf5")
print("Usage Training Mode: ipython main.py 0 parameters")
#TRAIN_DATASET = ['40_RZ_4983433_May-16-20-19-52']
#VAL_DATASET = ['45_RZ_5134712_May-18-14-14-00']
#TRAIN_DATASET = ['42_RZ_4988291_May-16-21-33-46']
#VAL_DATASET = ['44_RZ_5131746_May-18-13-25-32']
TRAIN_DATASET = ['36_RZ_4882422_May-15-16-08-32','38_RZ_4886422_May-15-17-15-33','39_RZ_4981421_May-16-19-40-17','43_RZ_5129700_May-18-12-52-16']
#TRAIN_DATASET = ['36_RZ_4882422_May-15-16-08-32']
VAL_DATASET = ['37_RZ_4883794_May-15-16-37-01']
#TRAIN_DATASET = ['47_KM_1535284_Jul-31-16-10-56']
#VAL_DATASET = ['48_KM_1537673_Jul-31-16-51-20']

#BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{40_RZ}tr_{45_RZ}val"
#BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{42_RZ}tr_{44_RZ}val"
#BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{36_RZ}tr_{37_RZ}val"
BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{36_38_39_43_RZ}tr_{37_RZ}val"
#BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{36_37_38_39_43_RZ}tr_{47_48_KM}val"
#BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{47_KM}tr_{48_KM}val"

LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
PREDICT_FILE_TRAIN = BASE_FILE_NAME + '-train-result'
PREDICT_FILE_VAL = BASE_FILE_NAME + '-val-result'
BATCH_SIZE = 50
num_epoch = 50
#lr = float(sys.argv[3])

MODEL_DIR = 'Seaquest_36-43_37_pastK+flow'
#MODEL_DIR = 'Breakout_42_44'
#MODEL_DIR = 'Seaquest_47_48'
#MODEL_DIR = 'Pacman_40_45'
#MODEL_DIR = 'Seaquest_RZ_KM'
#MODEL_DIR = 'Seaquest_36_37'

resume_model = False
predict_mode = int(sys.argv[1]) 
dropout = float(sys.argv[2])
k = int(sys.argv[3])
strides = int(sys.argv[4])
heatmap_shape = 84
SHAPE = (84,84,k) # height * width * channel This cannot read from file and needs to be provided here


if not predict_mode: # if train
    import input_utils as IU, misc_utils as MU
    expr = MU.ExprCreaterAndResumer(MODEL_DIR, postfix="2ch_dp" + str(dropout) + '_shape' + str(heatmap_shape)+'_k'+str(k)+'s'+str(strides))
    expr.redirect_output_to_logfile_if_not_on("eldar-11")
else:
    import all_py_files_snapshot.input_utils as IU, all_py_files_snapshot.misc_utils as MU

MU.save_GPU_mem_keras()
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
    # opt=K.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    
    model.compile(loss=MU.my_kld, optimizer=opt, metrics=[MU.NSS])
    
#time.sleep(5)
d=IU.DatasetWithHeatmap_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, heatmap_shape, GAZE_POS_ASC_FILE, k, strides, 0)
opf=IU.Dataset_OpticalFlow_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, k, strides, 0)

if not predict_mode: # if train
    expr.dump_src_code_and_model_def(sys.argv[0], model)

    model.fit([d.train_imgs, opf.train_flow], d.train_GHmap, BATCH_SIZE, epochs=num_epoch,
        validation_data=([d.val_imgs, opf.val_flow], d.val_GHmap, d.val_weight),
        shuffle=True, sample_weight = d.train_weight, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00001),
            MU.PrintLrCallback()])

    expr.save_weight_and_training_config_state(model)

    score = model.evaluate([d.val_imgs,opf.val_flow], d.val_GHmap, BATCH_SIZE, 0, sample_weight = d.val_weight)
    expr.printdebug("eval score:" + str(score))

elif predict_mode: # if predict
    model.load_weights(sys.argv[5])

    print "Evaluating model..."
    train_score = model.evaluate([d.train_imgs,opf.train_flow], d.train_GHmap, BATCH_SIZE, 0, d.train_weight)
    val_score = model.evaluate([d.val_imgs, opf.val_flow], d.val_GHmap, BATCH_SIZE, 0, d.val_weight)
    print "Train loss is:  " , train_score
    print "Val loss is: " , val_score

    print "Predicting results..."
    train_pred = model.predict(d.train_imgs, BATCH_SIZE) # [1] for prob
    val_pred = model.predict(d.val_imgs, BATCH_SIZE)
    print "Predicted."

    print "Converting predicted results into png files and save..."
    IU.save_heatmap_png_files(d.train_fid, train_pred, TRAIN_DATASET, 'saliency/')
    IU.save_heatmap_png_files(d.val_fid, val_pred, VAL_DATASET, 'saliency/')
    print "Done."