# compute the mean score of the saliency results

import numpy as np
import tarfile, sys, os, time, math
from scipy import misc
from sklearn import metrics
import copy as cp
sys.path.insert(0, '../shared') # After research, this is the best way to import a file in another dir
from scipy.stats import entropy, wasserstein_distance
from base_input_utils import read_gaze_data_asc_file, ForkJoiner, convert_gaze_pos_to_heap_map, preprocess_gaze_heatmap, rescale_and_clip_gaze_pos
import vip_constants as V

def computeNSS(saliency_map, gt_interest_points):
    if len(gt_interest_points) == 0:
        print("Warning: No gaze data for this frame!")
        return 2.0

    stddev = np.std(saliency_map)
    if (stddev > 0):
        sal = (saliency_map - np.mean(saliency_map)) / stddev
    else:
        sal = saliency_map

    score = np.mean([ sal[y][x] for x,y in gt_interest_points ])
    return score


def computeEMD(saliency_map, gt_saliency_map):
    if len(gt_saliency_map) == 0:
        print("Warning: gt_saliency_map length 0")
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    #saliency_map = saliency_map.flatten()
    #gt_saliency_map = gt_saliency_map.flatten()

    saliency_map = saliency_map / np.sum(saliency_map)
    d = cdist(gt_saliency_map, saliency_map)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() # / (len(gt_saliency_map)**2)
    #return wasserstein_distance(saliency_map, gt_saliency_map)

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

def computeRecall(saliency_map, gt_saliency_map):
    saliency_map = saliency_map.flatten()
    gt_saliency_map = gt_saliency_map.flatten()

    if len(gt_saliency_map) == 0:
        print("Empty ground truth saliency map, return a default value of 1")
        return 1.0
    pos = 0
    true_pos = 0
    for i in range(len(gt_saliency_map)):
        if gt_saliency_map[i] >= 0.01: 
            pos += 1
            if saliency_map[i] >= 0.01:
                true_pos += 1
    if pos == 0: score = 1
    else: score = float(true_pos) / pos
    return score

def computeAUC2(saliency_map, gt_saliency_map):
    gt_saliency_map = gt_saliency_map.flatten()
    max_indx = np.argmax(gt_saliency_map)
    mask_map = np.zeros(len(gt_saliency_map))
    mask_map[max_indx] = 1
    fpr, tpr, thresholds = metrics.roc_curve(mask_map, saliency_map.flatten())
    return metrics.auc(fpr, tpr)

def computeKL(saliency_map, gt_saliency_map):
    epsilon = 2.2204e-16 #MIT benchmark
    saliency_map = np.clip(saliency_map.flatten(), a_min=epsilon, a_max=None)
    saliency_map = saliency_map / np.sum(saliency_map)
    gt_saliency_map = gt_saliency_map.flatten()
    # gt_saliency_map = np.clip(gt_saliency_map.flatten(), epsilon, 1)

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
    A = np.asarray([1,0,1,0,0])
    GTA = np.asarray([1,1,0.000001,1,1])
    print(computeRecall(A, GTA))



