import numpy as np
import matplotlib.pyplot as plt
import copy

files = []
files.append("acc.txt")

datas = []
for file_ in files:
    temp_file = open(file_, 'r')
    data = []
    for i,line in enumerate(temp_file):
        if i == 0:
            angles = [float(x) for x in line.split()]
        if i == 1:
            fv_mean = [float(x) for x in line.split()]
        if i == 2:
            base_mean = [float(x) for x in line.split()]
        if i == 3:
            fv_ser = [float(x) for x in line.split()]
    temp_file.close()
    datas.append(copy.deepcopy(data))
# plot rewards
fig, (ax1) = plt.subplots(1,1)
ax1.errorbar(angles, fv_mean, yerr=fv_ser, fmt = 'o',color = 'b', ls = '--', lw = 2, ecolor = 'r')
#ax1.errorbar(angles, base_mean, yerr=0, color = 'b', ls = 'solid', lw = 2, label="Baseline")
handles, labels = ax1.get_legend_handles_labels()
#ax1.legend(handles, labels, loc = 'upper left', frameon = False)
plt.ylim((49,55))
plt.xlim((4,111))
plt.xlabel("Visual Angle (Degree)", fontsize = 17)
plt.ylabel("Prediction Accuracy (%)", fontsize = 17)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
#font = {'size': 32}
#plt.rc('font', **font)


plt.show()

#sample = []
#for i in range(20):
#    sample.append(20 * (i+1))
#
#plt.gcf().set_size_inches(5,5)
#plt.plot(sample,datas[0],'r',label = 'Foveated')
#plt.plot(sample,datas[1],'b',label = 'Baseline')
##plt.plot(sample,datas[2],'r',label = '$\lambda = 0.25$')
##plt.plot(sample,datas[3],'y',label = '$\lambda = 1$')
#
#legend = plt.legend(loc = 'upper right')
#plt.ylabel('Mean Squared Error of Rewards')
#plt.xlabel('Number of Samples')
##plt.yticks(fontsize = 'x-large')
##plt.xticks(fontsize = 'x-large')
#plt.savefig("sparse.png", dpi = 300)
##plt.show()
