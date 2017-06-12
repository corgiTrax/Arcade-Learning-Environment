import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

NUM_CLASSES=8
BASE_FILE_NAME = "/scratch/cluster/zharucs/dataset/cat{40_RZ}tr_{45_RZ}val"
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
SHAPE = (84,84,1) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE=100
num_epoch = 25
dropout=0.5
MODEL_DIR = 'Pacman_40_45'
resume_model = False
sigma=25
background=0

MU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

expr = MU.ExprCreaterAndResumer(MODEL_DIR,postfix="GaussConv_C81_BG%d_gCUR_gauss%d" % (background, sigma))
expr.redirect_output_to_logfile_if_not_on("eldar-11")

if resume_model:
    model = expr.load_weight_and_training_config_and_state()
    expr.printdebug("Checkpoint found. Resuming model at %s" % expr.dir_lasttime)
else:
    gaze_heatmaps = L.Input(shape=(SHAPE[0],SHAPE[1],1))
    g=gaze_heatmaps
    g=L.Conv2D(1, (81,81), strides=1, padding='same')(g)
    g=L.BatchNormalization()(g)
    g=L.Activation('relu')(g)
    x=L.Dropout(dropout)(x)

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
#    opt=K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#    opt=K.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=True)

    model.compile(loss={"prob":None, "logits": MU.loss_func},
                 optimizer=opt, metrics={"logits": MU.acc_})

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=input_utils.DatasetWithGaze(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, GAZE_POS_ASC_FILE, 
       bg_prob_density=background, gaussian_sigma=sigma)
# d=input_utils.DatasetWithGazeWindow(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, GAZE_POS_ASC_FILE, 
#       bg_prob_density=0.0, gaussian_sigma=gaussian_sigma, window_left_bound_ms=350, window_right_bound_ms=200)

model.fit([d.train_imgs, d.train_GHmap], d.train_lbl, BATCH_SIZE, epochs=num_epoch,
    validation_data=([d.val_imgs, d.val_GHmap], d.val_lbl),
    shuffle=True,verbose=2,
    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.001),
        MU.PrintLrCallback()])

expr.save_weight_and_training_config_state(model)

score = model.evaluate([d.val_imgs, d.val_GHmap], d.val_lbl, BATCH_SIZE, 0)
expr.printdebug("eval score:" + str(score))

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
