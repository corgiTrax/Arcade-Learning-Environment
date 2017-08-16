import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

NUM_CLASSES=8
BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{36_38_39_43_RZ}tr_{37_RZ}val"
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
SHAPE = (84,84,1) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE=100
num_epoch = 70
dropout = 0.25
MODEL_DIR = 'Seaquest_36-43_37'

MU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

expr = MU.ExprCreaterAndResumer(MODEL_DIR,postfix="predgaze_PreMul")
expr.redirect_output_to_logfile_if_not_on("eldar-11")

# Now build the model

gaze_heatmaps = L.Input(shape=(SHAPE[0],SHAPE[1],1))
g=gaze_heatmaps

imgs=L.Input(shape=SHAPE)
x=imgs
x=L.Multiply()([x,g])
x_intermediate=x
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
logits=L.Dense(NUM_CLASSES, name="logits")(x)
prob=L.Activation('softmax', name="prob")(logits)
model=Model(inputs=[imgs, gaze_heatmaps], outputs=[logits, prob, g, x_intermediate])

opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

model.compile(loss={"prob":None, "logits": MU.loss_func},
            optimizer=opt,metrics={"logits": MU.acc_})

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=input_utils.Dataset(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE)
d.train_imgs, d.val_imgs = d.train_imgs[8:], d.val_imgs[8:]
d.train_fid,  d.val_fid  = d.train_fid[8:], d.val_fid[8:]
d.train_lbl,  d.val_lbl  = d.train_lbl[8:], d.val_lbl[8:]
input_utils.load_predicted_gaze_heatmap_into_dataset_train_GHmap_val_GHmap__temp(
    '/scratch/cluster/zharucs/ale/gaze/Seaquest_36-43_37_pastK/79_pKf_3D_dp0.5_shape84_k8s1/cat{36_38_39_43_RZ}tr_{37_RZ}val-train-result.npz',
    '/scratch/cluster/zharucs/ale/gaze/Seaquest_36-43_37_pastK/79_pKf_3D_dp0.5_shape84_k8s1/cat{36_38_39_43_RZ}tr_{37_RZ}val-val-result.npz',
    dataset_obj=d)

model.fit([d.train_imgs, d.train_GHmap], d.train_lbl, BATCH_SIZE, epochs=num_epoch,
    validation_data=([d.val_imgs, d.val_GHmap], d.val_lbl),
    shuffle=True,verbose=2,
    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.001),
        MU.PrintLrCallback()])

expr.save_weight_and_training_config_state(model) # uncomment this line if you want to save model

score = model.evaluate([d.val_imgs, d.val_GHmap], d.val_lbl, BATCH_SIZE, 0)
expr.printdebug("eval score:" + str(score))

