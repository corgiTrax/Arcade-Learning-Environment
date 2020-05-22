import numpy as np
import sys
#sys.path.insert(0, '../shared')
from evaluation import computeCC

class Comparator:
    def __init__(self, rl_att_file, human_att_file):    
        # Process attention data for RL
        rl_atts = np.load(sys.argv[1], allow_pickle=True)
        self.rl_atts = rl_atts["att_gd"]
        # For experiment 1 this is 6x5x100x84x84; 2 and 5 are random seeds)
        self.rl_atts = self.rl_atts[5]
        self.rl_atts = np.mean(self.rl_atts, axis=0)
        print(self.rl_atts.shape)

        # Process attention data for human
        human_atts = np.load(human_att_file, allow_pickle=True)
        self.human_atts = human_atts["heatmap"]
        print(self.human_atts.shape)

    def calc_CC(self):
        if len(self.rl_atts) != len(self.human_atts):
            print("Error: length of human attention map and rl attention map do not agree, exit")
            sys.exit(1)
        cumCC = 0
        cumAUC = 0
        for i in range(len(self.rl_atts)):
            rl_att = self.rl_atts[i] / np.sum(self.rl_atts[i])
            cumCC += computeCC(rl_att, self.human_atts[i])
        self.CC = cumCC / len(self.rl_atts)
        print("CC:", self.CC)

sq_rl_data_cp = Comparator(sys.argv[1], sys.argv[2])
sq_rl_data_cp.calc_CC()


