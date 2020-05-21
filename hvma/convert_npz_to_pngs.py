import numpy as np
import sys
import matplotlib.pyplot as plt

data = np.load(sys.argv[1],allow_pickle=True)
imgs = data['raw']
print(imgs.shape)

PAST = 4
GAME = sys.argv[1]

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
