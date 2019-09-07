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

print("Usage: ipython main.py [PredictMode?]")
print("Usage Predict Mode: ipython main.py 1 parameters Model.hdf5")
print("Usage Training Mode: ipython main.py 0 parameters")

GAME_NAME = sys.argv[1]
if GAME_NAME == 'alien':
    VAL_DATASET = ['334_RZ_901798_Jun-18-18-50-40','367_RZ_2082483_Jul-02-10-49-27','421_RZ_2795983_Jul-10-17-03-49','491_RZ_3489522_Jul-18-17-41-40']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/alien/alien"  
    MODEL_DIR = "AAAI/alien"
elif GAME_NAME == 'asterix':
    VAL_DATASET = ['168_JAW_2357233_Mar-29-16-01-52','315_RZ_216627_Jun-10-20-31-25','502_RZ_3567229_Jul-19-15-14-30']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/asterix/asterix"  
    MODEL_DIR = "AAAI/asterix"
elif GAME_NAME == 'bank_heist':
    VAL_DATASET = ['355_RZ_1924894_Jun-30-15-02-13','380_RZ_2261928_Jul-04-12-39-29','417_RZ_2789701_Jul-10-15-15-49','470_RZ_3392231_Jul-17-14-40-04']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/bank_heist/bank_heist"  
    MODEL_DIR = "AAAI/bank_heist"
elif GAME_NAME == 'berzerk':
    VAL_DATASET = ['208_RZ_6886042_Jan-07-12-36-45','280_RZ_4950628_Apr-10-21-44-28','377_RZ_2182316_Jul-03-14-33-59','476_RZ_3400020_Jul-17-16-47-41']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/berzerk/berzerk"  
    MODEL_DIR = "AAAI/berzerk"
elif GAME_NAME == 'breakout':
    VAL_DATASET = ['140_JAW_3192484_Dec-13-13-20-26', '307_RZ_9590717_Jun-03-14-39-21', '92_RZ_3504740_Aug-23-11-27-56']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/breakout/breakout"  
    MODEL_DIR = "AAAI/breakout"

elif GAME_NAME == 'centipede':
    VAL_DATASET = ['204_RZ_4136256_Dec-06-16-48-50', '450_RZ_3221959_Jul-15-15-19-59', '69_RZ_2831643_Aug-15-16-16-35', '97_RZ_3586578_Aug-24-09-59-20']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/centipede/centipede"  
    MODEL_DIR = "AAAI/centipede"
elif GAME_NAME == 'demon_attack':
    VAL_DATASET = ['200_RZ_3969914_Dec-04-18-34-08', '256_LG_1308608_Feb-27-17-05-40', '262_LG_1478757_Mar-01-16-21-01', '495_RZ_3559787_Jul-19-13-10-28']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/demon_attack/demon_attack"  
    MODEL_DIR = "AAAI/demon_attack"
elif GAME_NAME == 'enduro':
    VAL_DATASET = ['146_JAW_3280425_Dec-14-13-41-19', '372_RZ_2174919_Jul-03-12-30-12', '473_RZ_3395627_Jul-17-15-34-33', '98_RZ_3588030_Aug-24-10-27-25']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/enduro/enduro"  
    MODEL_DIR = "AAAI/enduro"
elif GAME_NAME == 'freeway':
    VAL_DATASET = ['55_RZ_2464601_Aug-11-10-18-09', '617_RZ_5374717_Aug-09-13-19-21']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/freeway/freeway"  
    MODEL_DIR = "AAAI/freeway"
elif GAME_NAME == 'frostbite':
    VAL_DATASET = ['228_RZ_8272059_Jan-23-13-35-24', '354_RZ_1922766_Jun-30-14-26-45', '409_RZ_2690686_Jul-09-11-46-07', '492_RZ_3556049_Jul-19-12-11-20']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/frostbite/frostbite"  
    MODEL_DIR = "AAAI/frostbite"

elif GAME_NAME == 'hero':
    VAL_DATASET = ['270_RZ_3098729_Mar-20-11-21-51', '325_RZ_454040_Jun-13-14-28-10', '384_RZ_2266503_Jul-04-13-56-03', '482_RZ_3472304_Jul-18-12-52-35']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/hero/hero"  
    MODEL_DIR = "AAAI/hero"
elif GAME_NAME == 'montezuma_revenge':
    VAL_DATASET = ['324_RZ_452975_Jun-13-14-10-23', '371_RZ_2173469_Jul-03-12-05-16', '429_RZ_2945490_Jul-12-10-35-17', '493_RZ_3557734_Jul-19-12-36-14']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/montezuma_revenge/montezuma_revenge"  
    MODEL_DIR = "AAAI/montezuma_revenge"
elif GAME_NAME == 'ms_pacman':
    VAL_DATASET = ['137_KM_3115947_Dec-12-15-59-59', '273_RZ_3279899_Mar-22-13-39-02', '61_RZ_2737165_Aug-14-14-09-12', '91_RZ_3502739_Aug-23-10-43-13']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/ms_pacman/ms_pacman"  
    MODEL_DIR = "AAAI/ms_pacman"
elif GAME_NAME == 'name_this_game':
    VAL_DATASET = ['230_RZ_8457879_Jan-25-17-16-11', '353_RZ_1744577_Jun-28-13-01-09', '422_RZ_2797695_Jul-10-17-30-44', '500_RZ_3565079_Jul-19-14-39-15']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/name_this_game/name_this_game"  
    MODEL_DIR = "AAAI/name_this_game"
elif GAME_NAME == 'phoenix':
    VAL_DATASET = ['214_RZ_7226016_Jan-11-11-04-01', '357_RZ_1927863_Jun-30-15-51-50', '427_RZ_2884737_Jul-11-17-41-09', '477_RZ_3401134_Jul-17-17-06-25']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/phoenix/phoenix"  
    MODEL_DIR = "AAAI/phoenix"

elif GAME_NAME == 'riverraid':
    VAL_DATASET = ['102_RZ_3605715_Aug-24-15-16-09', '389_RZ_2352191_Jul-05-13-44-30', '485_RZ_3477674_Jul-18-14-23-17', '99_RZ_3590056_Aug-24-10-56-50']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/riverraid/riverraid"  
    B
    MODEL_DIR = "AAAI/riverraid"
elif GAME_NAME == 'road_runner':
    VAL_DATASET = ['303_RZ_9241473_May-30-13-38-34', '375_RZ_2179243_Jul-03-13-41-42', '425_RZ_2882662_Jul-11-17-05-01', '503_RZ_3568255_Jul-19-15-31-39']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/road_runner/road_runner"  
    MODEL_DIR = "AAAI/road_runner"
elif GAME_NAME == 'seaquest':
    VAL_DATASET = ['185_RZ_9437843_Jun-19-14-51-34', '407_RZ_2610962_Jul-08-13-37-00', '67_RZ_2823456_Aug-15-13-58-25', '87_RZ_3435454_Aug-22-16-01-22']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/seaquest/seaquest"  
    MODEL_DIR = "AAAI/seaquest"
elif GAME_NAME == 'space_invaders':
    VAL_DATASET = ['222_RZ_7759093_Jan-17-15-05-46', '302_RZ_9240251_May-30-13-19-31', '345_RZ_1491905_Jun-25-14-46-22', '497_RZ_3561920_Jul-19-13-46-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/space_invaders/space_invaders"  
    MODEL_DIR = "AAAI/space_invaders"
elif GAME_NAME == 'venture':
    VAL_DATASET = ['114_RZ_3870288_Aug-27-16-47-08', '155_KM_5791179_Jan-12-15-08-15', '462_RZ_3298909_Jul-16-12-44-38', '65_RZ_2812003_Aug-15-10-47-56']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/venture/venture"  
    MODEL_DIR = "AAAI/venture"

elif GAME_NAME == 'demon_attack_small':
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/demon_attack/demon_attack_small"  
    MODEL_DIR = "AAAI/demon_attack_small"
elif GAME_NAME == 'riverraid_small':
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/riverraid/riverraid_small"  
    MODEL_DIR = "AAAI/riverraid_small"
elif GAME_NAME == 'venture_small':
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/venture/venture_small"  
    MODEL_DIR = "AAAI/venture_small"

LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
AFFIX = '-image_past4'
PREDICT_FILE_VAL = BASE_FILE_NAME.split('/')[-1] + AFFIX
BATCH_SIZE = 32
num_epoch = 50

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
#    x=L.Activation('relu')(x)
#    x=L.BatchNormalization()(x)

    outputs = L.Activation(MU.my_softmax)(x)

    model=Model(inputs=inputs, outputs=outputs)

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #opt=K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) 

    model.compile(loss=MU.my_kld, optimizer=opt, metrics=[MU.NSS])
    #model.compile(loss=K.losses.kullback_leibler_divergence, optimizer=opt, metrics=[MU.NSS])
    
d=IU.DatasetWithHeatmap_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, heatmap_shape, GAZE_POS_ASC_FILE, k, stride)

if not predict_mode: # if train
    expr.dump_src_code_and_model_def(sys.argv[0], model)

    model.fit(d.train_imgs, d.train_GHmap, BATCH_SIZE, epochs=num_epoch,
        validation_data=(d.val_imgs, d.val_GHmap, d.val_weight),
        shuffle=True, sample_weight=d.train_weight, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00001),
            MU.BMU.PrintLrCallback()])
    
    expr.save_weight_and_training_config_state(model)

    score = model.evaluate(d.val_imgs, d.val_GHmap, BATCH_SIZE, 0, sample_weight=d.val_weight)
    expr.printdebug("eval score:" + str(score))

elif predict_mode: # if predict
    model.load_weights(sys.argv[4])

    print "Evaluating model..."
    #train_score = model.evaluate(d.train_imgs, d.train_GHmap, BATCH_SIZE, 0)
    #print "Train loss is:  " , train_score
    val_score = model.evaluate(d.val_imgs, d.val_GHmap, BATCH_SIZE, 0, d.val_weight)
    print "Val loss is: " , val_score

    print "Predicting results..."
    train_pred = model.predict(d.train_imgs, BATCH_SIZE) # [1] for prob
    val_pred = model.predict(d.val_imgs, BATCH_SIZE)
    print "Predicted."

    # Uncomment this block to save predicted gaze heatmap for visualization
    # print "Converting predicted results into png files and save..."
    # IU.save_heatmap_png_files(d.train_fid, train_pred, TRAIN_DATASET, 'saliency/')
    # IU.save_heatmap_png_files(d.val_fid, val_pred, VAL_DATASET, 'saliency/')
    # print "Done."

    print "Writing predicted gaze heatmap (train) into the npz file..."
    np.savez_compressed(BASE_FILE_NAME.split('/')[-1] + '-train' + AFFIX, fid=d.train_fid, heatmap=train_pred)
    print "Done. Output is:"
    print " %s" % BASE_FILE_NAME.split('/')[-1] + '-train' + AFFIX + '.npz'

    print "Writing predicted gaze heatmap (val) into the npz file..."
    np.savez_compressed(BASE_FILE_NAME.split('/')[-1] + '-val' + AFFIX, fid=d.val_fid, heatmap=val_pred)
    print "Done. Output is:"
    print " %s" % BASE_FILE_NAME.split('/')[-1] + '-val' + AFFIX + '.npz'
