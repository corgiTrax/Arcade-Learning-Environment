import numpy as np
import sys
import matplotlib.pyplot as plt

IS_ATT=True

data = np.load(sys.argv[1],allow_pickle=True)
GAME = sys.argv[2]

if IS_ATT == False:
    imgs = data['raw']
    print(imgs.shape)
    
    PAST = 4
    
    data_fname = GAME + ".txt"
    dfile = open(data_fname,'w')
    
    for i,img_stack16 in enumerate(imgs):
        dfs = []
        for j,img in enumerate(img_stack16):
            if j <= PAST:
                imgName = GAME + '/' + str(i) + "_" + str(PAST-j) + '.png'
                plt.imsave(imgName, img)
                dfs.append(imgName)
        for imgName in reversed(dfs):
            dfile.write(imgName)
            dfile.write("\n")
    dfile.close()
else:
    atts = data["att_gd"]
    print(atts.shape)

    atts = np.mean(atts, axis=(0,1))
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
        plt.imsave(GAME + "/" + str(i) + '.png', pic)
    print "Done."

