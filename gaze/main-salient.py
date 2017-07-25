import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{36_RZ}tr_{37_RZ}val"
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
SHAPE = (84,84,1) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE=100
num_epoch = 50
dropout = 0.0
#MODEL_DIR = 'Seaquest_36&38&39&43_37'
#MODEL_DIR = 'Breakout_42_44'
MODEL_DIR = 'Seaquest_36_37'
resume_model = False


MU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

expr = MU.ExprCreaterAndResumer(MODEL_DIR,postfix="baseline")
expr.redirect_output_to_logfile_if_not_on("eldar-11")

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
    
    conv3=L.Conv2D(80, (3,3), strides=2, padding='same')
    x = conv3(x)
    print conv3.output_shape
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
#    x=L.Conv2DTranspose(80, (3,3), strides = 2, padding = 'same', activation = 'relu')(x)
#    deconv1=L.Conv2DTranspose(40, (3,3), strides = 2, padding = 'same', activation = 'relu')
#    x = deconv1(x)
#    print deconv1.output_shape
#    deconv2=L.Conv2DTranspose(20, (4,4), strides = 2, padding = 'same', activation = 'relu')
#    x = deconv2(x)
#    print deconv2.output_shape
    last = L.Conv2DTranspose(1, (3,3), strides = 14, padding = 'same')
    outputs = last(x)
    print last.output_shape
#    embed()

#    x=L.Flatten()(x)
    
#    x=L.Dense(256, activation='relu')(x)
#    x=L.Dropout(dropout)(x)
#    logits=L.Dense(2, name="logits")(x)
#    x=L.Conv2DTranspose(filters = 1, kernel_size , strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    model=Model(inputs=inputs, outputs=outputs)

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #opt=K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt=K.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(loss='kullback_leibler_divergence', optimizer=opt, metrics=['mse'])

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=input_utils.DatasetWithHeatmap(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, GAZE_POS_ASC_FILE)
# print d.val_gaze[0][0],d.val_gaze[0][1]
#print("train: ", d.train_GHmap.shape)
#print(d.val_GHmap.shape)

model.fit(d.train_imgs, d.train_GHmap, BATCH_SIZE, epochs=num_epoch,
    validation_data=(d.val_imgs, d.val_GHmap),
    shuffle=True,verbose=2,
    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.001),
        MU.PrintLrCallback()])
#model.fit(d.train_imgs, d.train_gaze, BATCH_SIZE, epochs=num_epoch,
#    validation_data=(d.val_imgs, d.val_gaze),
#    shuffle=True,verbose=2,
#    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),  
#        MU.PrintLrCallback()])

expr.save_weight_and_training_config_state(model)

score = model.evaluate(d.val_imgs, d.val_GHmap, BATCH_SIZE, 0)
expr.printdebug("eval score:" + str(score))
