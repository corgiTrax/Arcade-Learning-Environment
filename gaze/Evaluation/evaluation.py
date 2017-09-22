# compute the mean score of the saliency results

import numpy as np
from input_utils import read_gaze_data_asc_file, ForkJoiner, convert_gaze_pos_to_heap_map, preprocess_gaze_heatmap, rescale_and_clip_gaze_pos
import tarfile, sys, os, time, math
from scipy import misc
from scipy.stats import entropy
import vip_constants as V
from sklearn import metrics
import copy as cp

def computeNSS(saliency_map, gt_interest_points):
    if len(gt_interest_points) == 0:
        print "Warning: No gaze data for this frame!"
        return 2.0

    stddev = np.std(saliency_map)
    if (stddev > 0):
        sal = (saliency_map - np.mean(saliency_map)) / stddev
    else:
        sal = saliency_map

    score = np.mean([ sal[y][x] for x,y in gt_interest_points ])
    return score

def computeCC(saliency_map, gt_saliency_map):
    saliency_map = saliency_map.flatten()
    gt_saliency_map = gt_saliency_map.flatten()

    if len(gt_saliency_map) == 0:
        return 1.0
    gt_sal = (gt_saliency_map - np.mean(gt_saliency_map)) / np.std(gt_saliency_map)
    stddev = np.std(saliency_map)
    if (stddev > 0):
        sal = (saliency_map - np.mean(saliency_map)) / stddev
        score = np.corrcoef([gt_sal, sal])[0][1]
    else:
        sal = saliency_map
        score = np.cov([gt_sal, sal])[0][1]

    return score

def computeKL(saliency_map, gt_saliency_map):
    epsilon = 1e-10
    saliency_map = np.clip(saliency_map.flatten(), epsilon, 1)
    gt_saliency_map = np.clip(gt_saliency_map.flatten(), epsilon, 1)

    return entropy(gt_saliency_map, saliency_map)

def computeAUC(saliency_map, fixationmap_gt):
    fixationmap_gt = np.clip(fixationmap_gt, 0, 1)
    fpr, tpr, thresholds = metrics.roc_curve(fixationmap_gt.flatten(), saliency_map.flatten())
    return metrics.auc(fpr, tpr)

def read_heatmap(heatmap_path):
    """
    Read the predicted heatmap from an npz file
    return a list of frame ids and corresponding heatmaps
    """
    data = np.load(heatmap_path)
    frameids = data['fid']
    heatmaps = data['heatmap']

    return frameids, heatmaps[...,0]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: %s dataset_gaze_asc_file result_saliency_npz_file" % sys.argv[0]
        sys.exit(0)

    GAZE_FILE = sys.argv[1]
    RESULT_SALIENCY = sys.argv[2]
    HEATMAP_SHAPE = (84,84)
    RESULT_FILE = "evaluation_result.txt"

    ############# result saliency map #######################
    t1 = time.time()
    print "Reading result saliency map..."
    frameid, heatmap_pred = read_heatmap(RESULT_SALIENCY)
    print "Done. Time spent to read saliency map: %.1fs" % (time.time()-t1)

    ############# ground truth ##############################
    t2 = time.time()
    print "Reading asc file..."
    gazepos, _ = read_gaze_data_asc_file(GAZE_FILE) # gazepos: {fid: [(1,1),(2,2)]}

    print "Processing gaze pos and converting to heatmap..."
    bad_count, tot_count = 0, 0
    heatmap_gt = []
    fixationmap_gt = []
    for fid_list in frameid:
        tmp = np.zeros([HEATMAP_SHAPE[0], HEATMAP_SHAPE[1], 1], dtype=np.float32)
        # exclude bad/no gaze
        fid = (fid_list[0],fid_list[1])
        if fid in gazepos and gazepos[fid]:
            tot_count += len(gazepos[fid])
            bad_gaze = convert_gaze_pos_to_heap_map(gazepos[fid], tmp)
            bad_count += bad_gaze
            if bad_gaze == 0:
                heatmap_gt.append(cp.deepcopy(tmp)) # Note that tmp is a pointer, you need to deep copy it 
                fixationmap_gt.append(cp.deepcopy(tmp))    

                for j in range(len(gazepos[fid])):
                    x = int(gazepos[fid][j][0]/V.SCR_W*HEATMAP_SHAPE[1])
                    y = int(gazepos[fid][j][1]/V.SCR_H*HEATMAP_SHAPE[0])
                    gazepos[fid][j] = (x,y)  
            else: # if there is bad gaze, discard this frame and all the gaze pos
                gazepos[fid] = []
                
    heatmap_gt = np.asarray(heatmap_gt, dtype=np.float32)
    fixationmap_gt = np.asarray(fixationmap_gt, dtype=np.int32)
    print "Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/tot_count, tot_count)    
    print "'Bad' means the gaze position is outside the 160*210 screen"

    sigmaH = 28.50 * HEATMAP_SHAPE[0] / V.SCR_H
    sigmaW = 44.58 * HEATMAP_SHAPE[1] / V.SCR_W
    heatmap_gt = preprocess_gaze_heatmap(heatmap_gt, sigmaH, sigmaW, 0)

    #heatmap_gt = fixationmap_gt #TODO just for test
    print "Normalizing the heat map..."
    for i in range(len(heatmap_gt)):
        SUM = heatmap_gt[i].sum()
        if SUM != 0:
            heatmap_gt[i] /= SUM
    print "Done. Time spent to read and process gaze data: %.1fs" % (time.time()-t2)

    ################# compute #################

    count = 0
    NSS_score = 0
    AUC_score = 0
    KL_score = 0
    CC_score = 0
    
    t3=time.time()
    print "Computing NSS AUC KL CC..."
    for (i,fid_list) in enumerate(frameid):
        print "\r%d/%d" % (i,len(frameid)),
        sys.stdout.flush()

        fid = (fid_list[0],fid_list[1])
        # exclude bad/no gaze
        if fid in gazepos and gazepos[fid]:
            NSS_score += computeNSS(heatmap_pred[i], gazepos[fid])

            AUC_score += computeAUC(heatmap_pred[i], fixationmap_gt[count,:,:,0])

            KL_score += computeKL(heatmap_pred[i], heatmap_gt[count,:,:,0])

            CC_score += computeCC(heatmap_pred[i], heatmap_gt[count,:,:,0])

            count += 1
    print "Done. Time spent to compute scores: %.1fs" % (time.time()-t3)

    print "NSS: %f" % (NSS_score*1.0 / count)
    print "AUC: %f" % (AUC_score*1.0 / count)
    print "KL: %f" % (KL_score*1.0 / count)
    print "CC: %f" % (CC_score*1.0 / count)
    
    print "Writing evaluation scores into file..."
    with open(RESULT_FILE, 'a') as f:
        f.write(RESULT_SALIENCY+'\n')
        f.write("NSS: %f\n" % (NSS_score*1.0 / count))
        f.write("AUC: %f\n" % (AUC_score*1.0 / count))
        f.write("KL: %f\n" % (KL_score*1.0 / count))
        f.write("CC: %f\n" % (CC_score*1.0 / count))
        f.write('\n')
    print "Done. Evaluation scores are written into file %s" % RESULT_FILE




