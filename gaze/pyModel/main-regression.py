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

BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{36_38_39_43_RZ}tr_{37_RZ}val"

LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
PREDICT_FILE_VAL = BASE_FILE_NAME + '-regression'
BATCH_SIZE = 50
num_epoch = 70

MODEL_DIR = 'Seaquest_regression'

resume_model = False
predict_mode = int(sys.argv[1]) 
dropout = float(sys.argv[2])
#lr = float(sys.argv[3])
#factor = float(sys.argv[4])
heatmap_shape = 84
k = 4
stride = 1
SHAPE = (84,84,k) # height * width * channel This cannot read from file and needs to be provided here


if not predict_mode: # if train
    import input_utils as IU, misc_utils as MU
    expr = MU.BMU.ExprCreaterAndResumer(MODEL_DIR, postfix='pKf_dp%.2f_k%ds%d' % (dropout,k,stride))
    expr.redirect_output_to_logfile_if_not_on("eldar-11")
else:
    import all_py_files_snapshot.input_utils as IU, all_py_files_snapshot.misc_utils as MU

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
    
    conv1=L.Conv2D(32, (8,8), strides=4, padding='same')
    x = conv1(x)
    print conv1.output_shape
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    conv2=L.Conv2D(64, (4,4), strides=2, padding='same')
    x = conv2(x)
    print conv2.output_shape
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
    conv3=L.Conv2D(64, (3,3), strides=1, padding='same')
    x = conv3(x)
    print conv3.output_shape
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)

    x=L.Dense(512, activation='relu')(x)
    x=L.Dropout(dropout)(x)
    logits=L.Dense(2, name="logits")(x)

    model=Model(inputs=inputs, outputs=logits)

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #opt=K.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt=K.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

    
d=IU.Dataset_PastKFrames(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, k, stride)

if not predict_mode: # if train
    expr.dump_src_code_and_model_def(sys.argv[0], model)

    model.fit(d.train_imgs, d.train_gaze, BATCH_SIZE, epochs=num_epoch,
        validation_data=(d.val_imgs, d.val_gaze, d.val_weight),
        shuffle=True, sample_weight=d.train_weight, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00001),
            MU.BMU.PrintLrCallback()])
    #model.fit(d.train_imgs, d.train_GHmap, BATCH_SIZE, epochs=num_epoch,
    #    validation_data=(d.val_imgs, d.val_GHmap),
    #    shuffle=True,verbose=2,
    #    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),  
    #        MU.BMU.PrintLrCallback()])

    expr.save_weight_and_training_config_state(model)

    score = model.evaluate(d.val_imgs, d.val_gaze, BATCH_SIZE, 0, sample_weight=d.val_weight)
    expr.printdebug("eval score:" + str(score))

elif predict_mode: # if predict
    model.load_weights(sys.argv[3])

    print "Evaluating model..."
    #train_score = model.evaluate(d.train_imgs, d.train_GHmap, BATCH_SIZE, 0)
    val_score = model.evaluate(d.val_imgs, d.val_gaze, BATCH_SIZE, 0, sample_weight=d.val_weight)
    #print "Train loss is:  " , train_score
    print "Val loss is: " , val_score[0]

    print "Predicting results..."
    #train_pred = model.predict(d.train_imgs, BATCH_SIZE) # [1] for prob
    val_pred = model.predict(d.val_imgs, BATCH_SIZE)
    print "Predicted."

    xy_str_val = []
    for i in range(d.val_size):
        xy_str_val.append('(%d,%d) %f %f' % (d.val_fid[i][0], d.val_fid[i][1], val_pred[i][0], val_pred[i][1]))
    
    print "Writing predicted results into the file..."
    with open(PREDICT_FILE_VAL, 'w') as f:
        f.write('\n'.join(xy_str_val))
        f.write('\n')
    print "Done. Output is:"
    print " %s" % PREDICT_FILE_VAL
    
    
