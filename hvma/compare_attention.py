import numpy as np
import sys
#sys.path.insert(0, '../shared')
from evaluation import computeCC, computeRecall, computeKL, computeEMD, computeAUC2

class Comparator:
    def __init__(self, rl_att_file, human_att_file):    
        # Process attention data for RL
        rl_atts = np.load(sys.argv[1], allow_pickle=True)
        print(rl_atts.keys())
        self.rl_atts = rl_atts["att_gd"]
        # For experiment 1 this is 6x5x100x84x84; 2 and 5 are random seeds)
        print("Shape of rl attention file before: ", self.rl_atts.shape)
        #self.rl_atts = self.rl_atts[4]
        #self.rl_atts = np.mean(self.rl_atts, axis=0)
        print("Shape after processing: ", self.rl_atts.shape)

        # Process attention data for human
        human_atts = np.load(human_att_file, allow_pickle=True)
        self.human_atts = human_atts["heatmap"]
        print(self.human_atts.shape)

    def compare(self):
        if len(self.rl_atts) != len(self.human_atts):
            print("Error: length of human attention map and rl attention map do not agree, exit")
            sys.exit(1)
        #CC
        cumCC = 0
        for i in range(len(self.rl_atts)):
            cumCC += computeCC(self.rl_atts[i], self.human_atts[i])
        self.CC = cumCC / len(self.rl_atts)
        print("CC:", self.CC)
        
#        #Recall
#        cumRecall = 0
#        for i in range(len(self.rl_atts)):
#            cumRecall += computeRecall(self.rl_atts[i], self.human_atts[i])
#        self.Recall = cumRecall / len(self.rl_atts)
#        print("Recall:", self.Recall)

        #AUC2
        cumAUC2 = 0
        for i in range(len(self.rl_atts)):
            cumAUC2 += computeAUC2(self.rl_atts[i], self.human_atts[i])
        self.AUC2 = cumAUC2 / len(self.rl_atts)
        print("AUC2:", self.AUC2)

        #KL
        cumKL = 0
        for i in range(len(self.rl_atts)):
            cumKL += computeKL(self.rl_atts[i], self.human_atts[i])
        self.KL = cumKL / len(self.rl_atts)
        print("KL:", self.KL)

#        #EMD
#        cumEMD = 0
#        for i in range(len(self.rl_atts)):
#            cumEMD += computeEMD(self.rl_atts[i], self.human_atts[i])
#        self.EMD = cumEMD / len(self.rl_atts)
#        print("EMD:", self.EMD)

sq_rl_data_cp = Comparator(sys.argv[1], sys.argv[2])
sq_rl_data_cp.compare()


