import numpy as np
import sys
import matplotlib.pyplot as plt
import os

#TODO
# 0,1,2 0:IMG 1:IMG_FAILURE 2:ATT
DATA_TYPE = 2
IMG_SIZE = 150

data = np.load(sys.argv[1],allow_pickle=True)
SAVE_FOLDER = "visualization_paper/" + sys.argv[2]

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

if DATA_TYPE == 0:
    from scipy import misc
    imgs = data['raw']
    print(imgs.shape)
    
    PAST = 0 #only get the last one
    
    for i,img_stack16 in enumerate(imgs):
        for j,img in enumerate(img_stack16):
            if j <= PAST:
                imgName = SAVE_FOLDER + '/' + str(i) + "_" + str(PAST-j) + '.png'
                #img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                img = misc.imresize(img, [IMG_SIZE,IMG_SIZE], interp='bilinear')
                plt.imsave(imgName, img)
elif DATA_TYPE == 1:
    from scipy import misc
    fail_key = data.keys()[0]
    raw_imgs = data[fail_key] 
    print(raw_imgs.shape)

    #TODO
    BEFORE_DEAD_IDX = 15
    PAST = 0
    imgs = []
    for i,raw_img_stack in enumerate(raw_imgs): # iterate over 100 images
        raw_img_stack16 = raw_img_stack[BEFORE_DEAD_IDX][0] #FAIL_IDX = 0
        for j,raw_img in enumerate(raw_img_stack16): # over 16 images
            if j <= PAST: # only take the past 4 frames
                imgName = SAVE_FOLDER + '/' + str(i) + "_" + str(PAST-j) + '.png'
                img = raw_img
                #img = np.dot(raw_img[...,:3], [0.2989, 0.5870, 0.1140])
                img = misc.imresize(img, [IMG_SIZE,IMG_SIZE], interp='bilinear')
                plt.imsave(imgName, img)
elif DATA_TYPE == 2:
    #TODO
    EXP = 40
    KEY = "att_gd"
    if EXP == 9: #human
        KEY = "heatmap" 
        atts = data[KEY]
        print(atts.shape)
    elif EXP == 5: #failures
        KEY = "failures"
        before_dead_indx = 15 #TODO
        atts = data[KEY]
        atts = atts[:,before_dead_indx,:,:] 
    elif EXP == 3:
        atts = data[KEY]
        print(atts.shape)
        chkpt = int(sys.argv[3]) #TODO
        atts_temp = []
        for i in range(5):
            idx = i * 9 + chkpt
            atts_temp.append(atts[idx])
        atts = np.mean(atts_temp, axis=0) # average over all seeds
    elif EXP == 30: #no train
        atts = data[KEY]
        print(atts.shape)
        atts = np.mean(atts, axis=0) # average over all seeds
    elif EXP == 6: # rl_att_highscore
        atts = data[KEY]
        print(atts.shape)
    elif EXP == 4: # discounts
        discount_index = int(sys.argv[3]) #TODO
        atts = data[KEY][discount_index]
        print(atts.shape) # should be 8(1) x5x100x84x84
        atts = np.mean(atts, axis=0) # average over all seeds
        print(atts.shape)
    elif EXP == 40: # discounts for freeway
        # For experiment 4 Freeway this is 5,5,5,5,5,5,4,4
        index = int(sys.argv[3]) #TODO
        if index <=5:
            start, end = 5*index, 5*index + 4
        elif index == 6:
            start, end = 30, 33
        elif index == 7:
            start, end = 34, 37
        atts = data[start:end+1]
        atts = np.mean(atts, axis=0) # average over all seeds
        print("Len after processing: ", len(atts))

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from scipy import misc
    # These may be needed for better visualization later
    from astropy.convolution import convolve
    from astropy.convolution.kernels import Gaussian2DKernel
    print "Converting predicted results into png files and save..."
    m = cm.ScalarMappable(cmap='jet')

    for i,att in enumerate(atts):
        temp = att
        pic = convolve(temp, Gaussian2DKernel(stddev=1))
        pic = m.to_rgba(temp)[:,:,:3]
        pic = misc.imresize(pic, [IMG_SIZE,IMG_SIZE], interp='bilinear')
        plt.imsave(SAVE_FOLDER + "/" + str(i) + '.png', pic)
    print "Done."

