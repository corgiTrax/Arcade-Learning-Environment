import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils as IU, misc_utils as MU
import ipdb
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

GAME_NAME = sys.argv[1]
if GAME_NAME == 'asterix_expert':
    VAL_DATASET = ['478_RZ_3402237_Jul-17-17-24-39','502_RZ_3567229_Jul-19-15-14-30']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/asterix/asterix_irl_expert"  
    MODEL_DIR = "IJCAI-base/asterix_expert"
elif GAME_NAME == 'berzerk_expert':
    VAL_DATASET = ['468_RZ_3315708_Jul-16-17-23-00','476_RZ_3400020_Jul-17-16-47-41']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/berzerk/berzerk_irl_expert"  
    MODEL_DIR = "IJCAI-base/berzerk_expert"
elif GAME_NAME == 'breakout_expert':
    VAL_DATASET = ['307_RZ_9590717_Jun-03-14-39-21', '472_RZ_3394545_Jul-17-15-16-29']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/breakout/breakout_irl_expert"  
    MODEL_DIR = "IJCAI-base/breakout_expert"
elif GAME_NAME == 'centipede_expert':
    VAL_DATASET = ['286_RZ_5620664_Apr-18-15-56-48', '450_RZ_3221959_Jul-15-15-19-59']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/centipede/centipede_irl_expert"  
    MODEL_DIR = "IJCAI-base/centipede_expert"
elif GAME_NAME == 'ms_pacman_expert':
    VAL_DATASET = ['271_RZ_3101375_Mar-20-12-03-54', '273_RZ_3279899_Mar-22-13-39-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/ms_pacman/ms_pacman_irl_expert"  
    MODEL_DIR = "IJCAI-base/ms_pacman_expert"
elif GAME_NAME == 'phoenix_expert':
    VAL_DATASET = ['305_RZ_9315734_May-31-10-16-17', '306_RZ_9589522_Jun-03-14-19-59']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/phoenix/phoenix_irl_expert"  
    MODEL_DIR = "IJCAI-base/phoenix_expert"
elif GAME_NAME == 'seaquest_expert':
    VAL_DATASET = ['351_RZ_1741925_Jun-28-12-12-46', '407_RZ_2610962_Jul-08-13-37-00']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/seaquest/seaquest_irl_expert"  
    MODEL_DIR = "IJCAI-base/seaquest_expert"
elif GAME_NAME == 'space_invaders_expert':
    VAL_DATASET = ['455_RZ_3228302_Jul-15-17-05-43', '497_RZ_3561920_Jul-19-13-46-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/space_invaders/space_invaders_irl_expert"  
    MODEL_DIR = "IJCAI-base/space_invaders_expert"


elif GAME_NAME == 'beam_rider_novice':
    VAL_DATASET = ['642_AS_5902997_Aug-15-16-04-39']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/beam_rider/beam_rider_irl_novice"  
    MODEL_DIR = "IJCAI-base/beam_rider_novice"
elif GAME_NAME == 'breakout_novice':
    VAL_DATASET = ['307_RZ_9590717_Jun-03-14-39-21', '472_RZ_3394545_Jul-17-15-16-29']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/breakout/breakout_irl_novice"  
    MODEL_DIR = "IJCAI-base/breakout_novice"
elif GAME_NAME == 'enduro_novice':
    VAL_DATASET = ['473_RZ_3395627_Jul-17-15-34-33', '498_RZ_3562958_Jul-19-14-03-19']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/enduro/enduro_irl_novice"  
    MODEL_DIR = "IJCAI-base/enduro_novice"
elif GAME_NAME == 'pong_novice':
    VAL_DATASET = ['580_AS_4765925_Aug-02-12-12-59']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/pong/pong_irl_novice"  
    MODEL_DIR = "IJCAI-base/pong_novice"
elif GAME_NAME == 'qbert_novice':
    VAL_DATASET = ['629_KD_5881383_Aug-15-10-03-52']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/qbert/qbert_irl_novice"  
    MODEL_DIR = "IJCAI-base/qbert_novice"
elif GAME_NAME == 'seaquest_novice':
    VAL_DATASET = ['351_RZ_1741925_Jun-28-12-12-46', '407_RZ_2610962_Jul-08-13-37-00']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/seaquest/seaquest_irl_novice"  
    MODEL_DIR = "IJCAI-base/seaquest_novice"
elif GAME_NAME == 'space_invaders_novice':
    VAL_DATASET = ['455_RZ_3228302_Jul-15-17-05-43', '497_RZ_3561920_Jul-19-13-46-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/space_invaders/space_invaders_irl_novice"  
    MODEL_DIR = "IJCAI-base/space_invaders_novice"

LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt'
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt'
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
SHAPE = (84,84,1) # height * width * channel. Cannot be read from file; Needs to be provided here
BATCH_SIZE = 50
num_epoch = 50
#MODEL_DIR = sys.argv[2]
dropout = float(sys.argv[2])
save_model = True #if '--save' in sys.argv else False # you can specify "--save" in argument

MU.BMU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

expr = MU.BMU.ExprCreaterAndResumer(MODEL_DIR,postfix="baseline_dr%s" % str(dropout))
expr.redirect_output_to_logfile_if_not_on("eldar-11")

if True: # I just want to indent
    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    x=L.Conv2D(32, (8,8), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    x=L.Conv2D(64, (4,4), strides=1, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    x=L.Conv2D(64, (3,3), strides=1, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)
    
    x=L.Dense(512, activation='relu')(x)
    x=L.Dropout(dropout)(x)
    logits=L.Dense(IU.V.NUM_ACTION, name="logits")(x)
    prob=L.Activation('softmax', name="prob")(logits)
    model=Model(inputs=inputs, outputs=[logits, prob])

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    model.compile(loss={"prob":None, "logits": MU.loss_func},
                 optimizer=opt,metrics=[K.metrics.sparse_categorical_accuracy])

#    model.compile(loss={"prob":None, "logits": MU.loss_func},
#                 optimizer=opt,metrics={"logits": MU.acc_})

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=IU.BIU.Dataset(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE)

model.fit(d.train_imgs, d.train_lbl, BATCH_SIZE, epochs=num_epoch,
    validation_data=(d.val_imgs, d.val_lbl, d.val_weight),
    shuffle=True, sample_weight=d.train_weight, verbose=2,
    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.001),
        MU.BMU.PrintLrCallback()])


score = model.evaluate(d.val_imgs, d.val_lbl, BATCH_SIZE, 0, sample_weight=d.val_weight)
expr.printdebug("eval score:" + str(score))

if save_model:
  expr.save_weight_and_training_config_state(model)
