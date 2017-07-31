import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb
import json
import sys

print("Usage: ipython main.py [PredictMode?]")
print("Usage Predict Mode: ipython main.py 1 parameters Model.hdf5 savingFileTag")
print("Usage Training Mode: ipython main.py 0 parameters")
#BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{36_RZ}tr_{37_RZ}val"
BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset_gaze/cat{36_38_39_43_RZ}tr_{37_RZ}val"
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
PREDICT_FILE_TRAIN = BASE_FILE_NAME + '-train-result'
PREDICT_FILE_VAL = BASE_FILE_NAME + '-val-result'
SHAPE = (84,84,1) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE = 50
num_epoch = 50
MODEL_DIR = 'Seaquest_36&38&39&43_37'
#MODEL_DIR = 'Breakout_42_44'
#MODEL_DIR = 'Seaquest_36_37'
resume_model = False
predict_mode = int(sys.argv[1]) 
dropout = float(sys.argv[2])
MU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

if not predict_mode: # if train
    expr = MU.ExprCreaterAndResumer(MODEL_DIR, postfix="salient" + str(dropout))
    expr.redirect_output_to_logfile_if_not_on("eldar-11")

if resume_model:
    model = expr.load_weight_and_training_config_and_state()
    expr.printdebug("Checkpoint found. Resuming model at %s" % expr.dir_lasttime)
else:
    ###############################
    # Architecture of the network #
    ###############################

    inputs=L.Input(shape=SHAPE)
    x=inputs # inputs is used by the line "Model(inputs, ... )" below
    x=L.Conv2D(32, (8,8), strides=4, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    x=L.Conv2D(64, (4,4), strides=2, padding='same')(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
    
    conv3=L.Conv2D(64, (3,3), strides=1, padding='same')
    x = conv3(x)
#    print conv3.output_shape
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)
   
    x = L.Flatten()(x)
    x = L.Dense(7056, activation = "softmax")(x)
    logits =  L.Reshape((84, 84, -1), name = "logits")(x)

    #model=Model(inputs=inputs, outputs=[logits,prob])
    model=Model(inputs=inputs, outputs=logits)
    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    # opt=K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt=K.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    
    model.compile(loss='kullback_leibler_divergence', optimizer=opt, metrics=['kullback_leibler_divergence'])
    #model.compile(loss={"logits": 'kullback_leibler_divergence', "prob":None}, optimizer=opt, metrics={"logits": 'mean_squared_error'})
    #model.compile(loss={"logits": 'mean_squared_error', "prob":None}, optimizer=opt, metrics={"logits": 'mean_squared_error'})
    #model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
d=input_utils.DatasetWithHeatmap(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, GAZE_POS_ASC_FILE)

if not predict_mode: # if train
    expr.dump_src_code_and_model_def(sys.argv[0], model)

    model.fit(d.train_imgs, d.train_GHmap, BATCH_SIZE, epochs=num_epoch,
        validation_data=(d.val_imgs, d.val_GHmap),
        shuffle=True,verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00001),
            MU.PrintLrCallback()])
    #model.fit(d.train_imgs, d.train_gaze, BATCH_SIZE, epochs=num_epoch,
    #    validation_data=(d.val_imgs, d.val_gaze),
    #    shuffle=True,verbose=2,
    #    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),  
    #        MU.PrintLrCallback()])

    expr.save_weight_and_training_config_state(model)

    score = model.evaluate(d.val_imgs, d.val_GHmap, BATCH_SIZE, 0)
    expr.printdebug("eval score:" + str(score))

elif predict_mode: # if predict
    model.load_weights(sys.argv[3])
    modelID = sys.argv[4] # whatever ID we want to name this

    print "Evaluating model..."
    train_score = model.evaluate(d.train_imgs, d.train_GHmap, BATCH_SIZE, 0)
    val_score = model.evaluate(d.val_imgs, d.val_GHmap, BATCH_SIZE, 0)
    print "Train loss is:  " , train_score
    print "Val loss is: " , val_score

    print "Predicting results..."
    train_pred = model.predict(d.train_imgs, BATCH_SIZE) # [1] for prob
    val_pred = model.predict(d.val_imgs, BATCH_SIZE)
    print "Predicted."

#    print "Converting result format..."
#    xy_str_train = []
#    for i in range(d.train_size):
#        string = '(%d,%d)' % (d.train_fid[i][0], d.train_fid[i][1])
#        for row in range(SHAPE[0]):
#            for col in range(SHAPE[1]):
#                string += ' %f' % train_pred[i][row][col]
#        xy_str_train.append(string)

#    xy_str_val = []
#    for i in range(d.val_size):
#        string = '(%d,%d)' % (d.val_fid[i][0], d.val_fid[i][1])
#        for row in range(SHAPE[0]):
#            for col in range(SHAPE[1]):
#                string += ' %f' % val_pred[i][row][col]
#        xy_str_train.append(string)

#    print "Writing predicted results into the file..."
#    with open(PREDICT_FILE_TRAIN, 'w') as f:
#        f.write('\n'.join(xy_str_train))
#        f.write('\n')
#    with open(PREDICT_FILE_VAL, 'w') as f:
#        f.write('\n'.join(xy_str_val))
#        f.write('\n')
#    print "Done. Outputs are:"
#    print " %s" % PREDICT_FILE_TRAIN
#    print " %s" % PREDICT_FILE_VAL

#    print "Converting to json format..."
#    fid = "BEFORE-FIRST-FRAME"
#    xy_str_train = {fid: []}
#    for i in range(d.train_size):
#        string = '(%d,%d)' % (d.train_fid[i][0], d.train_fid[i][1])
#        xy_str_train[string] = train_pred[i].tolist()

#    fid = "BEFORE-FIRST-FRAME"
#    xy_str_val = {fid: []}
#    for i in range(d.val_size):
#        string = '(%d,%d)' % (d.val_fid[i][0], d.val_fid[i][1])
#        xy_str_val[string] = val_pred[i].tolist()
#    print "Converted."

#    print "Writing predicted results into the file..."
#    with open(PREDICT_FILE_TRAIN, 'w') as f:
#        json.dump(xy_str_train, f)

#    with open(PREDICT_FILE_VAL, 'w') as f:
#        json.dump(xy_str_val, f)

    print "Writing predicted results into the npz file..."
    np.savez_compressed(PREDICT_FILE_TRAIN + modelID, fid=d.train_fid, heatmap=train_pred)
    np.savez_compressed(PREDICT_FILE_VAL + modelID, fid=d.val_fid, heatmap=val_pred)

    print "Done. Outputs are:"
    print " %s" % PREDICT_FILE_TRAIN + '.npz'
    print " %s" % PREDICT_FILE_VAL + '.npz'
