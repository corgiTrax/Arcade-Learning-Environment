import numpy as np
import sys
import matplotlib.pyplot as plt
import os

#TODO
IS_ATT=True

data = np.load(sys.argv[1],allow_pickle=True)
SAVE_FOLDER = sys.argv[2]

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

if IS_ATT == False:
    from scipy import misc
    imgs = data['raw']
    print(imgs.shape)
    
    PAST = 4
    
#    data_fname = SAVE_FOLDER + ".txt"
#    dfile = open(data_fname,'w')
    
    for i,img_stack16 in enumerate(imgs):
        dfs = []
        for j,img in enumerate(img_stack16):
            if j <= PAST:
                imgName = SAVE_FOLDER + '/' + str(i) + "_" + str(PAST-j) + '.png'
                img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                img = misc.imresize(img, [84,84], interp='bilinear')
                plt.imsave(imgName, img)
                dfs.append(imgName)
#        for imgName in reversed(dfs):
#            dfile.write(imgName)
#            dfile.write("\n")
    dfile.close()
else:
    #TODO
    KEY = "att_gd"
    # KEY = "heatmap"
    atts = data[KEY]
    print(atts.shape)
    #TODO
    atts = np.mean(atts[4], axis=0)

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    # These may be needed for better visualization later
    # from astropy.convolution import convolve
    # from astropy.convolution.kernels import Gaussian2DKernel
    print "Converting predicted results into png files and save..."
    m = cm.ScalarMappable(cmap='jet')

    for i,att in enumerate(atts):
        temp = att
        # pic = convolve(temp, Gaussian2DKernel(stddev=1))
        pic = m.to_rgba(temp)[:,:,:3]
        plt.imsave(SAVE_FOLDER + "/" + str(i) + '.png', pic)
    print "Done."

