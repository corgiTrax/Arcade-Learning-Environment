import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

# IMPORTANT NOTE: this file represents a bad-performing model, so its dependent code is deleted 
# from input_utils.py. However, a backup of the deleted code is copied at the end of this file.
# You can copy it back if you want to reuse it.

NUM_CLASSES=8
BASE_FILE_NAME = "/scratch/cluster/zhuode93/dataset/cat{36_FV}tr_{37_FV}val"
LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
k, stride = 1,1
SHAPE = (84,84,k) # height * width * channel This cannot read from file and needs to be provided here
BATCH_SIZE=100
num_epoch = 25
dropout = 0.25
MODEL_DIR = 'GazeExpr{36}tr_{37}val'

MU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

expr = MU.ExprCreaterAndResumer(MODEL_DIR,postfix="fov_CenterGaze_2x_k%ds%d" % (k, stride))
expr.redirect_output_to_logfile_if_not_on("eldar-11")

SHAPE_AFTER_CENTER_AT_GAZE_POS_PREPROCESSING = (SHAPE[0]*2,SHAPE[1]*2,SHAPE[2])
inputs=L.Input(shape=SHAPE_AFTER_CENTER_AT_GAZE_POS_PREPROCESSING)
x=inputs # inputs is used by the line "Model(inputs, ... )" below
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
model=Model(inputs=inputs, outputs=[logits, prob])

opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

model.compile(loss={"prob":None, "logits": MU.loss_func},
             optimizer=opt,metrics={"logits": MU.acc_})

expr.dump_src_code_and_model_def(sys.argv[0], model)

d=input_utils.DatasetCenteredAtLastGaze(LABELS_FILE_TRAIN, LABELS_FILE_VAL, SHAPE, GAZE_POS_ASC_FILE, K=k, stride=stride)
model.fit(d.train_imgs, d.train_lbl, BATCH_SIZE, epochs=num_epoch,
    validation_data=(d.val_imgs, d.val_lbl),
    shuffle=True,verbose=2,
    callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
        K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.001),
        MU.PrintLrCallback()])

# expr.save_weight_and_training_config_state(model) # uncomment this line if you want to save model

score = model.evaluate(d.val_imgs, d.val_lbl, BATCH_SIZE, 0)
expr.printdebug("eval score:" + str(score))



# ==== Begin deleted code in input_utils.py ==== (see explanation at the beginning of this file)

class DatasetCenteredAtLastGaze(Dataset):
    frameid2pos, frameid2action_notused = None, None

    def __init__(self, LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE, GAZE_POS_ASC_FILE, K=1, stride=1, before=0):
        super(DatasetCenteredAtLastGaze, self).__init__(LABELS_FILE_TRAIN, LABELS_FILE_VAL, RESIZE_SHAPE)
        self.RESIZE_SHAPE = RESIZE_SHAPE
        print "Reading gaze data ASC file..."
        self.frameid2pos, self.frameid2action_notused = read_gaze_data_asc_file(GAZE_POS_ASC_FILE)
        print "Running rescale_and_clip_gaze_pos_on_frameid2pos()..."
        self.rescale_and_clip_gaze_pos_on_frameid2pos(self.frameid2pos)

        print  "Running center_img_at_gaze() on train/val data..."
        rev = lambda (x,y): (y,x)
        last_gaze_of = lambda gaze_pos_list: rev(gaze_pos_list[-1]) if gaze_pos_list else (RESIZE_SHAPE[0]/2, RESIZE_SHAPE[1]/2)
        self.train_gaze_y_x = np.asarray([last_gaze_of(self.frameid2pos[fid]) for fid in self.train_fid])
        self.train_imgs = self.center_img_at_gaze(self.train_imgs, self.train_gaze_y_x)
        self.val_gaze_y_x = np.asarray([last_gaze_of(self.frameid2pos[fid]) for fid in self.val_fid])
        self.val_imgs = self.center_img_at_gaze(self.val_imgs, self.val_gaze_y_x)    

        print  "Making past-K-frame train/val data..."
        t1=time.time()
        self.train_imgs = transform_to_past_K_frames(self.train_imgs, K, stride, before)
        self.val_imgs = transform_to_past_K_frames(self.val_imgs, K, stride, before)
        # Trim labels. This is assuming the labels align with the training examples from the back!!
        # Could cause the model unable to train  if this assumption does not hold
        self.train_lbl = self.train_lbl[-self.train_imgs.shape[0]:]
        self.val_lbl = self.val_lbl[-self.val_imgs.shape[0]:]
        print "Time spent to transform train/val data to pask K frames: %.1fs" % (time.time()-t1)
  
    def rescale_and_clip_gaze_pos_on_frameid2pos(self, frameid2pos):
        bad_count, tot_count = 0, 0
        for fid in frameid2pos:
            gaze_pos_list = frameid2pos[fid]
            tot_count += len(gaze_pos_list)
            for (i, (x, y)) in enumerate(gaze_pos_list):
                isbad, newx, newy = rescale_and_clip_gaze_pos(x,y,self.RESIZE_SHAPE[0],self.RESIZE_SHAPE[1])
                bad_count += isbad
                gaze_pos_list[i] = (newx, newy)  
        print "Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/tot_count, tot_count)    
        print "'Bad' means the gaze position is outside the 160*210 screen"

    def center_img_at_gaze(self, frame_dataset, gaze_y_x_dataset, debug_plot_result=False):
        import tensorflow as tf, keras as K # don't move this to the top, as people who import this file might not have keras or tf
        import keras.layers as L
        gaze_y_x = L.Input(shape=(2,))
        imgs = L.Input(shape=(frame_dataset.shape[1:]))
        h, w = frame_dataset.shape[1:3]
        c_img = L.ZeroPadding2D(padding=(h,w))(imgs)
        c_img = L.Lambda(lambda x: tf.image.extract_glimpse(x, size=(2*h,2*w), offsets=(gaze_y_x+(h,w)), centered=False, normalized=False))(c_img)
        # c_img = L.Lambda(lambda x: tf.image.resize_images(x, size=(h,w)))(c_img)

        model=K.models.Model(inputs=[imgs, gaze_y_x], outputs=[c_img])
        model.compile(optimizer='rmsprop', # not used
          loss='categorical_crossentropy', # not used
          metrics=None)
        output=model.predict([frame_dataset, gaze_y_x_dataset], batch_size=500)
        if debug_plot_result:
            print r"""debug_plot_result is True. Entering IPython console. You can run:
            %matplotlib
            import matplotlib.pyplot as plt
            plt.style.use('grayscale')
            f, axarr = plt.subplots(1,2)
            rnd=np.random.randint(output.shape[0]); print "rand idx:", rnd
            axarr[0].imshow(frame_dataset[rnd,...,0])
            axarr[1].imshow(output[rnd,...,0])"""
            embed()

        return output
