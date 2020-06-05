import numpy as np
import sys
#sys.path.insert(0, '../shared')
from evaluation import computeCC, computeRecall, computeKL, computeEMD, computeAUC2, computeSIM, computeNSS, computeENT
from scipy.stats import sem
import copy as cp

DEBUG = True

class Comparator:
    def __init__(self, human_att_file):
        if DEBUG: print("Init...")
        # Process attention data for human
        human_atts = np.load(human_att_file, allow_pickle=True)
        self.human_atts = human_atts["heatmap"]
        if DEBUG: print(self.human_atts.shape)
    
    def init_data_exp0(self, rl_att_file):    
        # Process attention data for RL
        rl_atts = np.load(rl_att_file, allow_pickle=True)
        if DEBUG: print(rl_atts.keys())
        self.rl_atts = rl_atts["att_gd"]
        # For experiment 1 this is 100x84x84
        if DEBUG: print("Shape of rl attention file: ", self.rl_atts.shape)

    def init_data_exp1(self, rl_att_file):    
        # Process attention data for RL
        rl_atts = np.load(rl_att_file, allow_pickle=True)
        if DEBUG: print(rl_atts.keys())
        self.rl_atts = rl_atts["att_gd"]
        # For experiment 1 this is 6 (seeds) x5(reps) x100x84x84
        if DEBUG: print("Shape of rl attention file before: ", self.rl_atts.shape)
        self.rl_atts = np.mean(self.rl_atts, axis=(0,1)) # average over all seeds
        if DEBUG: print("Shape after processing: ", self.rl_atts.shape)

    def init_data_exp2(self, rl_att_file, clipped):    
        # Process attention data for RL
        self.rl_atts = np.load(rl_att_file, allow_pickle=True)
        # For experiment 2 this is 2[scaled=0, clipped=1] x6x100x84x84 or x5x100x84x84
        self.rl_atts = np.asarray(self.rl_atts[clipped])
        if DEBUG: print("Shape before processing: ", self.rl_atts.shape)
        self.rl_atts = np.mean(self.rl_atts, axis=0) # average over all seeds
        if DEBUG: print("Shape after processing: ", self.rl_atts.shape)

    def init_data_exp30(self, rl_att_file):    
        # Process attention data for RL
        rl_atts = np.load(rl_att_file, allow_pickle=True)
        if DEBUG: print(rl_atts.keys())
        self.rl_atts = rl_atts["att_gd"]
        # For experiment 3(no_train) this is 5x100x84x84
        if DEBUG: print("Shape of rl attention file before: ", self.rl_atts.shape)
        self.rl_atts = np.mean(self.rl_atts, axis=0) # average over all seeds
        if DEBUG: print("Shape after processing: ", self.rl_atts.shape)

    def init_data_exp3(self, rl_att_file, chkpt):    
        # Process attention data for RL
        rl_atts = np.load(rl_att_file, allow_pickle=True)
        if DEBUG: print(rl_atts.keys())
        rl_atts = rl_atts["att_gd"]
        # For experiment 3 this is 45 (5x9) x100x84x84
        if DEBUG: print("Shape of rl attention file before: ", rl_atts.shape)
        rl_atts_temp = []
        for i in range(5):
            idx = i * 9 + chkpt
            rl_atts_temp.append(cp.deepcopy(rl_atts[idx]))
        self.rl_atts = np.mean(rl_atts_temp, axis=0) # average over all seeds
        if DEBUG: print("Shape after processing: ", self.rl_atts.shape)

    def init_data_exp4(self, rl_att_file, index):    
        # Process attention data for RL
        rl_atts = np.load(rl_att_file, allow_pickle=True)
        if DEBUG: print(rl_atts.keys())
        self.rl_atts = rl_atts["att_gd"]
        # For experiment 4 this is 8 (discounts) x5(seeds) x100x84x84
        if DEBUG: print("Shape of rl attention file before: ", self.rl_atts.shape)
        self.rl_atts = self.rl_atts[index]
        self.rl_atts = np.mean(self.rl_atts, axis=0) # average over all seeds
        if DEBUG: print("Shape after processing: ", self.rl_atts.shape)

    def init_data_exp40(self, rl_att_file, index): #For freeway only 
        # Process attention data for RL
        self.rl_atts = np.load(rl_att_file, allow_pickle=True)
        # For experiment 4 Freeway this is 5,5,5,5,5,5,4,4
        if index <=5:
            start, end = 5*index, 5*index + 4
        elif index == 6:
            start, end = 30, 33
        elif index == 7:
            start, end = 34, 37
        self.rl_atts = self.rl_atts[start:end+1]
        if DEBUG: print("Shape after processing: ", len(self.rl_atts))
        self.rl_atts = np.mean(self.rl_atts, axis=0) # average over all seeds

    def init_data_exp5(self, rl_att_file, index):    
        # Process attention data for RL
        rl_atts = np.load(rl_att_file, allow_pickle=True)
        if DEBUG: print(rl_atts.keys())
        self.rl_atts = rl_atts["failures"]
        # For experiment 5 this is 100x10x84x84
        if DEBUG: print("Shape of rl attention file before: ", self.rl_atts.shape)
        self.rl_atts = self.rl_atts[:,index,:,:]
        if DEBUG: print("Shape after processing: ", self.rl_atts.shape)

    def init_data_exp6(self, rl_att_file):    
        # Process attention data for RL
        data = np.load(rl_att_file, allow_pickle=True)
        if DEBUG: print(data.keys())
        self.rl_atts = data["att_gd"]
        self.rl_actions = data["rl_actions"]
        self.human_actions = data["human_actions"]
        # For experiment 6 this is 100x10x84x84
        if DEBUG: print("Shape after processing: ", self.rl_atts.shape)

    def compare(self):
        if len(self.rl_atts) != len(self.human_atts):
            print("Error: length of human attention map and rl attention map do not agree, exit")
            sys.exit(1)
        #CC
        CCs= []
        for i in range(len(self.rl_atts)):
            CCs.append(computeCC(self.rl_atts[i], self.human_atts[i]))
        self.CC = np.mean(CCs)
        self.CCsem = sem(CCs)
        if DEBUG: print("CC+-sem: %.4f+-%.4f" % (self.CC, self.CCsem))
        
        #AUC2
        AUC2s = []
        for i in range(len(self.rl_atts)):
            AUC2s.append(computeAUC2(self.rl_atts[i], self.human_atts[i]))
        self.AUC2 = np.mean(AUC2s)
        self.AUC2sem = sem(AUC2s)
        if DEBUG: print("AUC2+-sem: %.4f+-%.4f" % (self.AUC2, self.AUC2sem))

        #KL
        KLs = []
        for i in range(len(self.rl_atts)):
            KLs.append(computeKL(self.rl_atts[i], self.human_atts[i]))
        self.KL = np.mean(KLs)
        self.KLsem = sem(KLs)
        if DEBUG: print("KL+-sem: %.4f+-%.4f" % (self.KL, self.KLsem))

        #SIM
        SIMs = []
        for i in range(len(self.rl_atts)):
            SIMs.append(computeSIM(self.rl_atts[i], self.human_atts[i]))
        self.SIM = np.mean(SIMs)
        self.SIMsem = sem(SIMs)
        if DEBUG: print("SIM+-sem: %.4f+-%.4f" % (self.SIM, self.SIMsem))

#        # entropy of rl_atts
#        ENTs = []
#        for i in range(len(self.rl_atts)):
#            ENTs.append(computeENT(self.rl_atts[i]))
#        self.ENT = np.mean(ENTs)
#        self.ENTsem = sem(ENTs)
#        if DEBUG: print("ENT+-sem: %.4f+-%.4f" % (self.ENT, self.ENTsem))
#        #Recall
#        Recalls = []
#        for i in range(len(self.rl_atts)):
#            Recalls.append(computeRecall(self.rl_atts[i], self.human_atts[i]))
#        self.Recall = np.mean(Recalls)
#        self.Recallsem = sem(Recalls)
#        if DEBUG: print("Recall+-sem: %.4f+-%.4f" % (self.Recall, self.Recallsem))
#         #NSS
#        NSSs = []
#        for i in range(len(self.rl_atts)):
#            NSSs.append(computeNSS(self.rl_atts[i], self.human_atts[i]))
#        self.NSS = np.mean(NSSs)
#        self.NSSsem = sem(NSSs)
#        if DEBUG: print("NSS+-sem: %.4f+-%.4f" % (self.NSS, self.NSSsem))

    def print_results(self):
        print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f"
        % (self.CC, self.CCsem, self.AUC2, self.AUC2sem, self.KL, self.KLsem, self.SIM, self.SIMsem))

    def compare_exp6(self):
        if len(self.rl_atts) != len(self.human_atts):
            print("Error: length of human attention map and rlattention map do not agree, exit")
            sys.exit(1)
        #CC
        CCs_same, CCs_diff = [], []
        self.same_count = 0
        for i in range(len(self.rl_atts)):
            if self.rl_actions[i] == self.human_actions[i]:
                CCs_same.append(computeCC(self.rl_atts[i], self.human_atts[i]))
                self.same_count += 1
            else:
                CCs_diff.append(computeCC(self.rl_atts[i], self.human_atts[i]))
        self.CC_same, self.CCsem_same = np.mean(CCs_same), sem(CCs_same)
        self.CC_diff, self.CCsem_diff = np.mean(CCs_diff), sem(CCs_diff)
        if DEBUG: print("Same action CC+-sem: %.4f+-%.4f" % (self.CC_same, self.CCsem_same))
        if DEBUG: print("Diff action CC+-sem: %.4f+-%.4f" % (self.CC_diff, self.CCsem_diff))
        if DEBUG: print("Same action number: %.0f" % self.same_count)
        #AUC2
        AUC2s_same, AUC2s_diff = [], []
        same_count = 0
        for i in range(len(self.rl_atts)):
            if self.rl_actions[i] == self.human_actions[i]:
                AUC2s_same.append(computeAUC2(self.rl_atts[i], self.human_atts[i]))
                same_count += 1
            else:
                AUC2s_diff.append(computeAUC2(self.rl_atts[i], self.human_atts[i]))
        self.AUC2_same, self.AUC2sem_same = np.mean(AUC2s_same), sem(AUC2s_same)
        self.AUC2_diff, self.AUC2sem_diff = np.mean(AUC2s_diff), sem(AUC2s_diff)
        if DEBUG: print("Same action AUC2+-sem: %.4f+-%.4f" % (self.AUC2_same, self.AUC2sem_same))
        if DEBUG: print("Diff action AUC2+-sem: %.4f+-%.4f" % (self.AUC2_diff, self.AUC2sem_diff))
        #KL
        KLs_same, KLs_diff = [], []
        same_count = 0
        for i in range(len(self.rl_atts)):
            if self.rl_actions[i] == self.human_actions[i]:
                KLs_same.append(computeKL(self.rl_atts[i], self.human_atts[i]))
                same_count += 1
            else:
                KLs_diff.append(computeKL(self.rl_atts[i], self.human_atts[i]))
        self.KL_same, self.KLsem_same = np.mean(KLs_same), sem(KLs_same)
        self.KL_diff, self.KLsem_diff = np.mean(KLs_diff), sem(KLs_diff)
        if DEBUG: print("Same action KL+-sem: %.4f+-%.4f" % (self.KL_same, self.KLsem_same))
        if DEBUG: print("Diff action KL+-sem: %.4f+-%.4f" % (self.KL_diff, self.KLsem_diff))
        #SIM
        SIMs_same, SIMs_diff = [], []
        same_count = 0
        for i in range(len(self.rl_atts)):
            if self.rl_actions[i] == self.human_actions[i]:
                SIMs_same.append(computeSIM(self.rl_atts[i], self.human_atts[i]))
                same_count += 1
            else:
                SIMs_diff.append(computeSIM(self.rl_atts[i], self.human_atts[i]))
        self.SIM_same, self.SIMsem_same = np.mean(SIMs_same), sem(SIMs_same)
        self.SIM_diff, self.SIMsem_diff = np.mean(SIMs_diff), sem(SIMs_diff)
        if DEBUG: print("Same action SIM+-sem: %.4f+-%.4f" % (self.SIM_same, self.SIMsem_same))
        if DEBUG: print("Diff action SIM+-sem: %.4f+-%.4f" % (self.SIM_diff, self.SIMsem_diff))

    def print_results_exp6(self):
        print(self.same_count)
        print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f"\
        % (self.CC_same, self.CCsem_same, self.AUC2_same, self.AUC2sem_same, self.KL_same, self.KLsem_same, self.SIM_same, self.SIMsem_same))
        print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f"\
        % (self.CC_diff, self.CCsem_diff, self.AUC2_diff, self.AUC2sem_diff, self.KL_diff, self.KLsem_diff, self.SIM_diff, self.SIMsem_diff))

EXP = int(sys.argv[3])
cprt = Comparator(sys.argv[1])
if EXP == 0: # A single prediction 
    cprt.init_data_exp0(sys.argv[2])
    cprt.compare()
    cprt.print_results()
elif EXP == 1: # average all seeds all repeats
    cprt.init_data_exp1(sys.argv[2])
    cprt.compare()
    cprt.print_results()
elif EXP == 2: # clipped or not
    for i in range(2):
        cprt.init_data_exp2(sys.argv[2], i)
        cprt.compare()
        cprt.print_results()
elif EXP == 30: # checkpoints, this is for exp3-notraining ones 
    cprt.init_data_exp30(sys.argv[2])
    cprt.compare()
    cprt.print_results()
elif EXP == 3: # checkpoints
    for i in range(9):
        if DEBUG: print("checkpoint index"+str(i)+"=======")
        cprt.init_data_exp3(sys.argv[2], i)
        cprt.compare()
        cprt.print_results()
elif EXP == 4: # discounts
    for i in range(8):
        if DEBUG: print("discount index"+str(i)+"=======")
        cprt.init_data_exp4(sys.argv[2], i)
        cprt.compare()
        cprt.print_results()
elif EXP == 40: # discounts
    for i in range(8):
        if DEBUG: print("discount index"+str(i)+"=======")
        cprt.init_data_exp40(sys.argv[2], i)
        cprt.compare()
        cprt.print_results()
elif EXP == 5: # failures
    index = int(sys.argv[4]) # before_dead_index
    cprt.init_data_exp5(sys.argv[2], index)
    cprt.compare()
    cprt.print_results()
elif EXP == 6: # action_dependent analysis 
    cprt.init_data_exp6(sys.argv[2])
    cprt.compare_exp6()
    cprt.print_results_exp6()



