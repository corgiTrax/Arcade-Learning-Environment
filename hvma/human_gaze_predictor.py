import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
import sys
sys.path.insert(0, '../gaze')
import time
import copy as cp


class Human_Gaze_Predictor:
    def __init__(self, gaze_model_file, mean_file, data_file):
        self.gaze_model_file = gaze_model_file
        self.mean_file = mean_file
        self.game_name = gaze_model_file.split(".")[0]
        # data_file is a npz file
        self.data_file = data_file
        # img_name_file is a list of images, which is not in used now
        # self.img_name_file = img_name_file
        # Constants
        self.k = 4
        self.stride = 1
        self.img_shape = 84

    def init_model(self):
        # Imported from ../gaze/
        import input_utils as IU, misc_utils as MU
        MU.BMU.save_GPU_mem_keras()
        MU.keras_model_serialization_bug_fix()

        # Constants
        SHAPE = (self.img_shape,self.img_shape,self.k) # height * width * channel This cannot read from file and needs to be provided here
        dropout = 0.0
        ###############################
        # Architecture of the network #
        ###############################
        inputs=L.Input(shape=SHAPE)
        x=inputs # inputs is used by the line "Model(inputs, ... )" below
        
        conv1=L.Conv2D(32, (8,8), strides=4, padding='valid')
        x = conv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv2=L.Conv2D(64, (4,4), strides=2, padding='valid')
        x = conv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv3=L.Conv2D(64, (3,3), strides=1, padding='valid')
        x = conv3(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid')
        x = deconv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
    
        deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid')
        x = deconv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)         
    
        deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid')
        x = deconv3(x)
    
        outputs = L.Activation(MU.my_softmax)(x)
        self.model=Model(inputs=inputs, outputs=outputs)
        opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=MU.my_kld, optimizer=opt, metrics=[MU.NSS])
        
        print("Loading model weights from %s" % self.gaze_model_file)
        self.model.load_weights(self.gaze_model_file)
        print("Loaded.")
  
    def init_data(self):
        from scipy import misc
        data = np.load(self.data_file, allow_pickle=True)
        raw_imgs = data["raw"] # 100 x 16 x 210 x 160 x 3
        
        PAST = 4
        self.imgs = []
        for i,raw_img_stack16 in enumerate(raw_imgs):
            stack = [] # 4 x 84 x 84
            for j,raw_img in enumerate(raw_img_stack16):
                if j < PAST: # only take the past 4 frames
                    # raw_img.shape is 210 x 160 x 3
                    img = np.dot(raw_img[...,:3], [0.2989, 0.5870, 0.1140])
                    img = misc.imresize(img, [self.img_shape,self.img_shape], interp='bilinear')
                    img = img.astype(np.float16) / 255.0 # normalize image to [0,1]
                    stack.append(cp.deepcopy(img))
                    stack.reverse() # Note: LB stored imgs in npz in backward order
            self.imgs.append(cp.deepcopy(stack))
        self.imgs = np.asarray(self.imgs)
        self.imgs = self.imgs.transpose([0,2,3,1]) # 100 x 84 x 84 x 4
        # standardize
        mean = np.load(self.mean_file)
        self.imgs -= mean

    def predict_and_save(self):
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        # These may be needed for better visualization later
        # from astropy.convolution import convolve
        # from astropy.convolution.kernels import Gaussian2DKernel
    
        print "Predicting results..."
        self.preds = self.model.predict(self.imgs) 
        print "Predicted."
    
        # Uncomment this block to save predicted gaze heatmap for visualization
        print "Converting predicted results into png files and save..."
        m = cm.ScalarMappable(cmap='jet')
        for i,pred in enumerate(self.preds):
            temp = pred[:,:,0]
            # pic = convolve(temp, Gaussian2DKernel(stddev=1))
            pic = m.to_rgba(temp)[:,:,:3]
            plt.imsave(self.game_name + "/" + str(i) + '.png', pic)
        print "Done."
    
    #    print "Writing predicted gaze heatmap (train) into the npz file..."
    #    np.savez_compressed(self.game_name, heatmap=train_pred)
    #    print "Done. Output is:"
    #    print " %s" % BASE_FILE_NAME.split('/')[-1] + '-train' + AFFIX + '.npz'
    
if __name__ == "__main__":
    pass

# Not in use; TODO for future: this seems to be past4 not including the current one, may be problematic
def transform_to_past_K_frames(original, K, stride, before):
    newdat = []
    for i in range(before+K*stride, len(original)):
        # transform the shape (K, H, W, CH) into (H, W, CH*K)
        cur = original[i-before : i-before-K*stride : -stride] # using "-stride" instead of "stride" lets the indexing include i rather than exclude i
        cur = cur.transpose([1,2,3,0])
        cur = cur.reshape(cur.shape[0:2]+(-1,))
        newdat.append(cur)
        if len(newdat)>1: assert (newdat[-1].shape == newdat[-2].shape) # simple sanity check
    newdat_np = np.array(newdat)
    return newdat_np

#    def init_data_from_imgs(self):
#        from scipy import misc
#        imgs = [None] * 5 #TODO
#        imlistfile = open(self.img_name_file)
#        for i,line in enumerate(imlistfile):
#            imgfile = line.strip("\n")
#            img = misc.imread(imgfile, 'Y') # 'Y': grayscale  
#            img = misc.imresize(img, [self.img_shape,self.img_shape], interp='bilinear')
#            img = np.expand_dims(img, axis=2)
#            img = img.astype(np.float16) / 255.0 # normalize image to [0,1]
#            imgs[i] = img
#        # standardize
#        mean = np.load(self.mean_file)
#        imgs -= mean
#        imgs = np.asarray(imgs)
#        print(imgs.shape)
#        self.imgs = transform_to_past_K_frames(imgs, self.k, self.stride, 0)
#        print(self.imgs.shape)
#
