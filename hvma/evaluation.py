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

BG = 1.0/(84*84)
def preprocess(smap):
    smap = smap.flatten()
    smap = smap / np.sum(smap)
    return smap
def preprocess2(smap):
    smap = smap.flatten()
    smap -= BG
    smap = np.clip(smap, a_min=0, a_max=None)
    smap = smap / np.sum(smap)
    return smap

def computeNSS(saliency_map, gt_saliency_map):
    max_indx = np.unravel_index(np.argmax(gt_saliency_map, axis=None), gt_saliency_map.shape)
    #mask_map = np.zeros(shape = gt_saliency_map.shape)
    #mask_map[max_indx] = 1

    stddev = np.std(saliency_map)
    if (stddev > 0):
        sal = (saliency_map - np.mean(saliency_map)) / stddev
    else:
        sal = saliency_map

    score = np.mean([ sal[x][y] for x,y in [max_indx] ])
    return score

def computeEMD(saliency_map, gt_saliency_map):
    if len(gt_saliency_map) == 0:
        print("Warning: gt_saliency_map length 0")
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    saliency_map = saliency_map / np.sum(saliency_map)
    d = cdist(gt_saliency_map, saliency_map)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() # / (len(gt_saliency_map)**2)
    #return wasserstein_distance(saliency_map, gt_saliency_map)

def computeCC(saliency_map, gt_saliency_map):
    saliency_map = preprocess(saliency_map)
    gt_saliency_map = preprocess(gt_saliency_map)
    score = np.corrcoef([gt_saliency_map, saliency_map])[0][1]
    return score

def computeSIM(saliency_map, gt_saliency_map):
    saliency_map = preprocess(saliency_map)
    gt_saliency_map = preprocess(gt_saliency_map)
    sim = 0
    for i in range(len(gt_saliency_map)):
        sim += min(saliency_map[i], gt_saliency_map[i])
    return sim

def computeRecall(saliency_map, gt_saliency_map):
    saliency_map = preprocess(saliency_map)
    gt_saliency_map = preprocess2(gt_saliency_map)

    pos = 0
    true_pos = 0
    for i in range(len(gt_saliency_map)):
        if gt_saliency_map[i] > BG: 
            pos += 1
            if saliency_map[i] > BG:
                true_pos += 1
    if pos == 0: score = 1
    else: score = float(true_pos) / pos
    return score

def computeAUC2(saliency_map, gt_saliency_map):
    saliency_map = preprocess(saliency_map)
    gt_saliency_map = preprocess(gt_saliency_map)
    max_indx = np.argmax(gt_saliency_map)
    mask_map = np.zeros(len(gt_saliency_map))
    mask_map[max_indx] = 1
    
    fpr, tpr, thresholds = metrics.roc_curve(mask_map, saliency_map)
    return metrics.auc(fpr, tpr)

def computeKL(saliency_map, gt_saliency_map):
    epsilon = 2.2204e-16 #MIT benchmark
    saliency_map = preprocess(saliency_map)
    saliency_map = np.clip(saliency_map.flatten(), a_min=epsilon, a_max=None)
    saliency_map = saliency_map / np.sum(saliency_map)
    gt_saliency_map = preprocess2(gt_saliency_map)

    return entropy(gt_saliency_map, saliency_map)

def computeENT(saliency_map):
    saliency_map = preprocess2(saliency_map)
    return entropy(saliency_map)

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



