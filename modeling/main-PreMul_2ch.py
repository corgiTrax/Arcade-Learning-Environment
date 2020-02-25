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
    MODEL_DIR = "IJCAI/asterix_expert"
elif GAME_NAME == 'berzerk_expert':
    VAL_DATASET = ['468_RZ_3315708_Jul-16-17-23-00','476_RZ_3400020_Jul-17-16-47-41']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/berzerk/berzerk_irl_expert"  
    MODEL_DIR = "IJCAI/berzerk_expert"
elif GAME_NAME == 'breakout_expert':
    VAL_DATASET = ['307_RZ_9590717_Jun-03-14-39-21', '472_RZ_3394545_Jul-17-15-16-29']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/breakout/breakout_irl_expert"  
    MODEL_DIR = "IJCAI/breakout_expert"
elif GAME_NAME == 'centipede_expert':
    VAL_DATASET = ['286_RZ_5620664_Apr-18-15-56-48', '450_RZ_3221959_Jul-15-15-19-59']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/centipede/centipede_irl_expert"  
    MODEL_DIR = "IJCAI/centipede_expert"
elif GAME_NAME == 'ms_pacman_expert':
    VAL_DATASET = ['271_RZ_3101375_Mar-20-12-03-54', '273_RZ_3279899_Mar-22-13-39-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/ms_pacman/ms_pacman_irl_expert"  
    MODEL_DIR = "IJCAI/ms_pacman_expert"
elif GAME_NAME == 'phoenix_expert':
    VAL_DATASET = ['305_RZ_9315734_May-31-10-16-17', '306_RZ_9589522_Jun-03-14-19-59']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/phoenix/phoenix_irl_expert"  
    MODEL_DIR = "IJCAI/phoenix_expert"
elif GAME_NAME == 'seaquest_expert':
    VAL_DATASET = ['351_RZ_1741925_Jun-28-12-12-46', '407_RZ_2610962_Jul-08-13-37-00']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/seaquest/seaquest_irl_expert"  
    MODEL_DIR = "IJCAI/seaquest_expert"
elif GAME_NAME == 'space_invaders_expert':
    VAL_DATASET = ['455_RZ_3228302_Jul-15-17-05-43', '497_RZ_3561920_Jul-19-13-46-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/space_invaders/space_invaders_irl_expert"  
    MODEL_DIR = "IJCAI/space_invaders_expert"


elif GAME_NAME == 'beam_rider_novice':
    VAL_DATASET = ['642_AS_5902997_Aug-15-16-04-39']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/beam_rider/beam_rider_irl_novice"  
    MODEL_DIR = "IJCAI/beam_rider_novice"
elif GAME_NAME == 'breakout_novice':
    VAL_DATASET = ['307_RZ_9590717_Jun-03-14-39-21', '472_RZ_3394545_Jul-17-15-16-29']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/breakout/breakout_irl_novice"  
    MODEL_DIR = "IJCAI/breakout_novice"
elif GAME_NAME == 'enduro_novice':
    VAL_DATASET = ['473_RZ_3395627_Jul-17-15-34-33', '498_RZ_3562958_Jul-19-14-03-19']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/enduro/enduro_irl_novice"  
    MODEL_DIR = "IJCAI/enduro_novice"
elif GAME_NAME == 'pong_novice':
    VAL_DATASET = ['580_AS_4765925_Aug-02-12-12-59']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/pong/pong_irl_novice"  
    MODEL_DIR = "IJCAI/pong_novice"
elif GAME_NAME == 'qbert_novice':
    VAL_DATASET = ['629_KD_5881383_Aug-15-10-03-52']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/qbert/qbert_irl_novice"  
    MODEL_DIR = "IJCAI/qbert_novice"
elif GAME_NAME == 'seaquest_novice':
    VAL_DATASET = ['351_RZ_1741925_Jun-28-12-12-46', '407_RZ_2610962_Jul-08-13-37-00']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/seaquest/seaquest_irl_novice"  
    MODEL_DIR = "IJCAI/seaquest_novice"
elif GAME_NAME == 'space_invaders_novice':
    VAL_DATASET = ['455_RZ_3228302_Jul-15-17-05-43', '497_RZ_3561920_Jul-19-13-46-02']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/space_invaders/space_invaders_irl_novice"  
    MODEL_DIR = "IJCAI/space_invaders_novice"


elif GAME_NAME == 'seaquest-small':
    VAL_DATASET = ['185_RZ_9437843_Jun-19-14-51-34', '407_RZ_2610962_Jul-08-13-37-00']
    BASE_FILE_NAME = "/scratch/cluster/zharucs/data/seaquest/seaquest-small"  
    MODEL_DIR = "IJCAI/seaquest"


LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt'
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt'
pastK=4
pKf_or_cur = '-image_past4.npz' #TODO image+opf
PRED_GAZE_FILE_TRAIN = BASE_FILE_NAME + '-train' + pKf_or_cur
PRED_GAZE_FILE_VAL = BASE_FILE_NAME + '-val' + pKf_or_cur
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
SHAPE = (84,84,1) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE = 50
num_epoch = 50
#MODEL_DIR = sys.argv[2]
dropout = float(sys.argv[2])
resume_model = False
sigma_multiplier = float(sys.argv[3])
save_model = True #if '--save' in sys.argv else False

MU.BMU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()
expr = MU.BMU.ExprCreaterAndResumer(MODEL_DIR,postfix="PreMul_2ch_dr%.1f_img_sigma%.3f" % (dropout, sigma_multiplier)) #TODO img+opf
print sys.argv

if True: # I just want to indent
    gaze_heatmaps = L.Input(shape=(SHAPE[0],SHAPE[1],1))
    g=gaze_heatmaps
    g=L.BatchNormalization()(g)

    imgs=L.Input(shape=SHAPE)
    x=imgs
    x=L.Multiply()([x,g])
    x_intermediate=x
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
# ============================ channel 2 ============================
    orig_x=imgs
    orig_x=L.Conv2D(32, (8,8), strides=2, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)
    orig_x=L.Dropout(dropout)(orig_x)

    orig_x=L.Conv2D(64, (4,4), strides=1, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)
    orig_x=L.Dropout(dropout)(orig_x)

    orig_x=L.Conv2D(64, (3,3), strides=1, padding='same')(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)

    x=L.Average()([x,orig_x])
    x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)
    x=L.Dense(512, activation='relu')(x)
    x=L.Dropout(dropout)(x)
    logits=L.Dense(IU.V.NUM_ACTION, name="logits")(x)
    prob=L.Activation('softmax', name="prob")(logits)

    model=Model(inputs=[imgs, gaze_heatmaps], outputs=[logits, prob, g, x_intermediate])
    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #TODO: here we use logits for loss, what about probability?
    model.compile(loss={"prob":None, "logits": MU.loss_func},
                optimizer=opt,metrics=[K.metrics.sparse_categorical_accuracy])

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=IU.BIU.Dataset(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE)
IU.load_predicted_gaze_heatmap_into_dataset_train_GHmap_val_GHmap(
    PRED_GAZE_FILE_TRAIN, PRED_GAZE_FILE_VAL, d, pastK)

model.fit([d.train_imgs, d.train_GHmap], d.train_lbl, BATCH_SIZE, epochs=num_epoch,
    validation_data=([d.val_imgs, d.val_GHmap], d.val_lbl),
    shuffle=True,verbose=2,
    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.001),
        MU.BMU.PrintLrCallback()])

score = model.evaluate([d.val_imgs, d.val_GHmap], d.val_lbl, BATCH_SIZE, 0)
expr.printdebug("eval score:" + str(score))

if save_model:
  expr.save_weight_and_training_config_state(model) # uncomment this line if you want to save model

# add 'embed()' between "d=input_utils.Dataset..." and "model.fit()", then copy & paste the following to visualize 2 intermediate layers
# res=model.predict([d.val_imgs, d.val_GHmap])
# # Depending on the specfic model definition, you might need to change 'model.layers[0]', model.layers[3]' to something else
# # In general, look at https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
# g2convout=K.backend.function([model.layers[0].input,K.backend.learning_phase()],[model.layers[3].output]) 
# idx=1200
# f,axarr=plt.subplots(1,5)
# axarr[0].imshow(d.val_imgs[idx,...,0])
# axarr[1].imshow(d.val_GHmap[idx,...,0])
# axarr[2].imshow(g2convout([d.val_GHmap[idx].reshape(1,84,84,1),1])[0].reshape(84,84))
# axarr[3].imshow(res[2][idx,...,0])
# axarr[4].imshow(res[3][idx,...,0])
