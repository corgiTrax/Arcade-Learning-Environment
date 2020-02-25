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
if GAME_NAME == 'asterix_expert':
    VAL_DATASET = ['478_RZ_3402237_Jul-17-17-24-39','502_RZ_3567229_Jul-19-15-14-30']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/asterix/asterix_irl_expert"  
    MODEL_DIR = "IJCAI-s221-sm12/asterix_expert"
elif GAME_NAME == 'berzerk_expert':
    VAL_DATASET = ['468_RZ_3315708_Jul-16-17-23-00','476_RZ_3400020_Jul-17-16-47-41']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/berzerk/berzerk_irl_expert"  
    MODEL_DIR = "IJCAI-s221-sm12/berzerk_expert"
elif GAME_NAME == 'breakout_expert':
    VAL_DATASET = ['307_RZ_9590717_Jun-03-14-39-21', '472_RZ_3394545_Jul-17-15-16-29']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/breakout/breakout_irl_expert"  
    MODEL_DIR = "IJCAI-s221-sm12/breakout_expert"
elif GAME_NAME == 'centipede_expert':
    VAL_DATASET = ['286_RZ_5620664_Apr-18-15-56-48', '450_RZ_3221959_Jul-15-15-19-59']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/centipede/centipede_irl_expert"  
    MODEL_DIR = "IJCAI-s221-sm12/centipede_expert"
elif GAME_NAME == 'ms_pacman_expert':
    VAL_DATASET = ['271_RZ_3101375_Mar-20-12-03-54', '273_RZ_3279899_Mar-22-13-39-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/ms_pacman/ms_pacman_irl_expert"  
    MODEL_DIR = "IJCAI-s221-sm12/ms_pacman_expert"
elif GAME_NAME == 'phoenix_expert':
    VAL_DATASET = ['305_RZ_9315734_May-31-10-16-17', '306_RZ_9589522_Jun-03-14-19-59']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/phoenix/phoenix_irl_expert"  
    MODEL_DIR = "IJCAI-s221-sm12/phoenix_expert"
elif GAME_NAME == 'seaquest_expert':
    VAL_DATASET = ['351_RZ_1741925_Jun-28-12-12-46', '407_RZ_2610962_Jul-08-13-37-00']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/seaquest/seaquest_irl_expert"  
    MODEL_DIR = "IJCAI-s221-sm12/seaquest_expert"
elif GAME_NAME == 'space_invaders_expert':
    VAL_DATASET = ['455_RZ_3228302_Jul-15-17-05-43', '497_RZ_3561920_Jul-19-13-46-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/space_invaders/space_invaders_irl_expert"  
    MODEL_DIR = "IJCAI-s221-sm12/space_invaders_expert"


elif GAME_NAME == 'beam_rider_novice':
    VAL_DATASET = ['642_AS_5902997_Aug-15-16-04-39']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/beam_rider/beam_rider_irl_novice"  
    MODEL_DIR = "IJCAI-s221-sm12/beam_rider_novice"
elif GAME_NAME == 'breakout_novice':
    VAL_DATASET = ['307_RZ_9590717_Jun-03-14-39-21', '472_RZ_3394545_Jul-17-15-16-29']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/breakout/breakout_irl_novice"  
    MODEL_DIR = "IJCAI-s221-sm12/breakout_novice"
elif GAME_NAME == 'enduro_novice':
    VAL_DATASET = ['473_RZ_3395627_Jul-17-15-34-33', '498_RZ_3562958_Jul-19-14-03-19']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/enduro/enduro_irl_novice"  
    MODEL_DIR = "IJCAI-s221-sm12/enduro_novice"
elif GAME_NAME == 'pong_novice':
    VAL_DATASET = ['580_AS_4765925_Aug-02-12-12-59']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/pong/pong_irl_novice"  
    MODEL_DIR = "IJCAI-s221-sm12/pong_novice"
elif GAME_NAME == 'qbert_novice':
    VAL_DATASET = ['629_KD_5881383_Aug-15-10-03-52']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/qbert/qbert_irl_novice"  
    MODEL_DIR = "IJCAI-s221-sm12/qbert_novice"
elif GAME_NAME == 'seaquest_novice':
    VAL_DATASET = ['351_RZ_1741925_Jun-28-12-12-46', '407_RZ_2610962_Jul-08-13-37-00']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/seaquest/seaquest_irl_novice"  
    MODEL_DIR = "IJCAI-s221-sm12/seaquest_novice"
elif GAME_NAME == 'space_invaders_novice':
    VAL_DATASET = ['455_RZ_3228302_Jul-15-17-05-43', '497_RZ_3561920_Jul-19-13-46-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/space_invaders/space_invaders_irl_novice"  
    MODEL_DIR = "IJCAI-s221-sm12/space_invaders_novice"

LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
PREDICT_FILE_VAL = BASE_FILE_NAME.split('/')[-1] + '-image_curframe'
BATCH_SIZE = 50
num_epoch = 50

resume_model = False
predict_mode = int(sys.argv[2]) 
dropout = float(sys.argv[3])
gaze_weight = float(sys.argv[4])
sigma_multiplier=float(sys.argv[5])
heatmap_shape = 16 #TODO for conv1,2,3, if stride=2,2,1/2,1,1 then this is 39,18,16/39,36,34
SHAPE = (84,84,1) # height * width * channel This cannot read from file and needs to be provided here

import input_utils as IU, misc_utils as MU
if not predict_mode: # if train
    expr = MU.BMU.ExprCreaterAndResumer(MODEL_DIR, postfix='current_dp%.2f_gw%.5f_sigma%.1f' % (dropout, gaze_weight, sigma_multiplier))
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
    
    conv1=L.Conv2D(32, (8,8), strides=2, padding='valid')
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
    logits=L.Dense(IU.V.NUM_ACTION, name="logits")(y)
    prob=L.Activation('softmax', name="prob")(logits)
   
    #GCL otuput
    conv4 = L.Conv2D(1, (1,1), strides=1, padding='valid')
    z = conv4(x)
    print conv4.output_shape
    conv_output = L.Activation(MU.my_softmax, name="gaze_cvg")(z)

    # No normalization version
    #conv_output = L.Conv2D(1, (1,1), strides=1, padding='valid', name="gaze_cvg")(x)

    model=Model(inputs=inputs, outputs=[conv_output, logits, prob])

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #opt=K.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 

    model.compile(optimizer=opt, \
    loss={"gaze_cvg": MU.my_gcl_modifiedKL, "logits": MU.loss_func, "prob": None},\
    loss_weights={"logits":1 - gaze_weight, "gaze_cvg": gaze_weight},\
    metrics={"logits": K.metrics.sparse_categorical_accuracy, "gaze_cvg": MU.my_gcl_simplifiedKL})

d=IU.DatasetWithHeatmap(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, heatmap_shape, GAZE_POS_ASC_FILE, sigma_multiplier)

if not predict_mode: # if train
    expr.dump_src_code_and_model_def(sys.argv[0], model)

    # TODO unclear why need two weights
    model.fit(d.train_imgs, {"logits": d.train_lbl, "gaze_cvg": d.train_GHmap}, BATCH_SIZE, epochs=num_epoch,
        validation_data=(d.val_imgs, {"logits": d.val_lbl, "gaze_cvg": d.val_GHmap}, {"logits": d.val_weight, "gaze_cvg": d.val_weight}),
        shuffle=True, sample_weight={"logits": d.train_weight, "gaze_cvg": d.train_weight}, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00001),
            MU.BMU.PrintLrCallback()])
    
    expr.save_weight_and_training_config_state(model)

    score = model.evaluate(d.val_imgs, {"logits": d.val_lbl, "gaze_cvg": d.val_GHmap}, BATCH_SIZE, 0, sample_weight={"logits": d.val_weight, "gaze_cvg": d.val_weight})
    expr.printdebug("eval score:" + str(score))

elif predict_mode: # if predict
    model.load_weights(sys.argv[6])

    print "Evaluating model..."
    #train_score = model.evaluate(d.train_imgs, d.train_GHmap, BATCH_SIZE, 0)
    val_score = model.evaluate(d.val_imgs, {"logits": d.train_lbl, "gaze_cvg": d.train_GHmap}, BATCH_SIZE, 0, d.val_weight)
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
