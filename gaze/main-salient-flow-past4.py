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

GAME_NAME = sys.argv[1]
if GAME_NAME == 'seaquest':
    VAL_DATASET = ['70_RZ_2898339_Aug-16-10-58-55','75_RZ_3006069_Aug-17-16-46-05']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/seaquest_{70_75}val"  
    MODEL_DIR = "OpticalFlow/Seaquest"
elif GAME_NAME == 'mspacman':
    VAL_DATASET = ['71_RZ_2901714_Aug-16-11-54-21','76_RZ_3010789_Aug-17-18-01-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/mspacman_{71_76}val"  
    MODEL_DIR = "OpticalFlow/Mspacman"
elif GAME_NAME == 'centipede':
    VAL_DATASET = ['78_RZ_3068875_Aug-18-10-10-05','80_RZ_3084132_Aug-18-14-23-21']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/centipede_{78_80}val"  
    MODEL_DIR = "OpticalFlow/Centipede"
elif GAME_NAME == 'freeway':
    VAL_DATASET = ['72_RZ_2903977_Aug-16-12-25-04']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/freeway_{72}val"  
    MODEL_DIR = "OpticalFlow/Freeway"
elif GAME_NAME == 'venture':
    VAL_DATASET = ['100_RZ_3592991_Aug-24-11-44-38','101_RZ_3603032_Aug-24-14-31-37']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/venture_{100_101}val"  
    MODEL_DIR = "OpticalFlow/Venture"    
elif GAME_NAME == 'riverraid':
    VAL_DATASET = ['95_RZ_3522292_Aug-23-16-09-15','99_RZ_3590056_Aug-24-10-56-50']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/riverraid_{95_99}val"  
    MODEL_DIR = "OpticalFlow/Riverraid"
elif GAME_NAME == 'enduro':
    VAL_DATASET = ['98_RZ_3588030_Aug-24-10-27-25','103_RZ_3608911_Aug-24-16-17-04']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/enduro_{98_103}val"  
    MODEL_DIR = "OpticalFlow/Enduro"
elif GAME_NAME == 'breakout':
    VAL_DATASET = ['92_RZ_3504740_Aug-23-11-27-56']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/breakout_{92}val"  
    MODEL_DIR = "OpticalFlow/Breakout" 

LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
PREDICT_FILE_VAL = BASE_FILE_NAME.split('/')[-1] + '-flow_past4'
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


if not predict_mode: # if train
    import input_utils as IU, misc_utils as MU
    expr = MU.ExprCreaterAndResumer(MODEL_DIR, postfix='pKf_dp%.2f_k%ds%d' % (dropout,k,stride))
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

    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    
    conv1=L.Conv2D(32, (8,8), strides=4, padding='valid')
    x = conv1(x)
    print conv1.output_shape
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
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

    outputs = L.Activation(MU.my_softmax)(x)

    model=Model(inputs=inputs, outputs=outputs)

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #opt=K.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 

    model.compile(loss=MU.my_kld, optimizer=opt, metrics=[MU.NSS])
    
d=IU.DatasetWithHeatmap_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, heatmap_shape, GAZE_POS_ASC_FILE, k, stride)
of=IU.Dataset_OpticalFlow_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, k, stride)


if not predict_mode: # if train
    expr.dump_src_code_and_model_def(sys.argv[0], model)

    model.fit(of.train_flow, d.train_GHmap, BATCH_SIZE, epochs=num_epoch,
        validation_data=(of.val_flow, d.val_GHmap, d.val_weight),
        shuffle=True, sample_weight=d.train_weight, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00001),
            MU.PrintLrCallback()])
    
    expr.save_weight_and_training_config_state(model)

    score = model.evaluate(of.val_flow, d.val_GHmap, BATCH_SIZE, 0, sample_weight=d.val_weight)
    expr.printdebug("eval score:" + str(score))

elif predict_mode: # if predict
    model.load_weights(sys.argv[4])

    print "Evaluating model..."
    #train_score = model.evaluate(d.train_imgs, d.train_GHmap, BATCH_SIZE, 0)
    val_score = model.evaluate(of.val_flow, d.val_GHmap, BATCH_SIZE, 0, d.val_weight)
    #print "Train loss is:  " , train_score
    print "Val loss is: " , val_score

    print "Predicting results..."
    #train_pred = model.predict(d.train_imgs, BATCH_SIZE) # [1] for prob
    val_pred = model.predict(of.val_flow, BATCH_SIZE)
    print "Predicted."

    #print "Converting predicted results into png files and save..."
    #IU.save_heatmap_png_files(d.train_fid, train_pred, TRAIN_DATASET, 'saliency/')
    #IU.save_heatmap_png_files(d.val_fid, val_pred, VAL_DATASET, 'saliency/')
    #print "Done."

    print "Writing predicted results into the npz file..."
    np.savez_compressed(PREDICT_FILE_VAL, fid=d.val_fid, heatmap=val_pred)
    print "Done. Output is:"
    print " %s" % PREDICT_FILE_VAL+'.npz'
