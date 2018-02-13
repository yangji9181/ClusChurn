import math
import numpy as np
import sys
import csv

def h_utils(w, n):
    if 0 == w:
        return 0
    return -w * math.log(float(w) / n)

def cover_entropy (xs, n):
    tot_ent = 0.0
    for x in xs:
        tot_ent += h_utils(sum(x), n) + h_utils(n - sum(x), n)
    return tot_ent

def calc_modified_conditional_matrix(pre_ys, true_ys, n):
    results_0 = np.zeros((len(pre_ys), len(true_ys)))
    results_1 = np.zeros((len(true_ys), len(pre_ys)))
    for ind_p in range(0, len(pre_ys)):
        pre_y = pre_ys[ind_p]
        for ind_t in range(0, len(true_ys)):
            true_y = true_ys[ind_t]
            a = sum([ 0 == (py + ty) for (py, ty) in zip(pre_y, true_y)])
            d = sum([ 2 == (py + ty) for (py, ty) in zip(pre_y, true_y)])
            b = sum(true_y) - d
            c = sum(pre_y) - d
            t1 = h_utils(a, n) + h_utils(d, n)
            t2 = h_utils(b, n) + h_utils(c, n)
            t3 = h_utils(c + d, n) + h_utils(a + b, n)
            t4 = h_utils(b + d, n) + h_utils(a + c, n)
            if t1 >= t2:
                results_0[ind_p][ind_t] = t1 + t2 - t3
                results_1[ind_t][ind_p] = t1 + t2 - t4
            else:
                results_0[ind_p][ind_t] = t3
                results_1[ind_t][ind_p] = t4
    return results_0, results_1

def nmi_community(pre_ys, true_ys):
    """
    Normalized Mutual Information to evaluate overlapping community finding algorithms
    """
    n = len(pre_ys[0])
    hx = cover_entropy(pre_ys, n)
    hy = cover_entropy(true_ys, n)
    hxy, hyx = calc_modified_conditional_matrix(pre_ys, true_ys, n)
    hxy = sum([min(hxy_x) for hxy_x in hxy])
    hyx = sum([min(hyx_y) for hyx_y in hyx])
    return 0.5 * (hx + hy - hxy - hyx) / max(hx, hy)

def f1_pair(pred_y, true_y):
    """calculate f1 score for a pair of communities (predicted and ground truth)

    args: 
        pred_y (N * 1): binary array, 1 means the corresponding instance belongs to predicted community
        true_y (N * 1): binary array, 1 means the corresponding instance belongs to golden community
    """
    corrected = sum([ 2 == (py + ty) for (py, ty) in zip(pred_y, true_y)])
    if 0 == corrected:
        return 0, 0, 0
    precision = float(corrected) / sum(pred_y)
    recall = float(corrected) / sum(true_y)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score, precision, recall

def f1_community(pre_ys, true_ys):
    """calculate f1 score for two sets of communities (predicted and ground truth)

    args: 
        pred_ys (k * N): 
        true_ys (l * N):
    """
    tot_size = 0
    tot_fscore = 0.0
    for pre_y in pre_ys:
        cur_size = sum(pre_y)
        tot_size += cur_size
        tot_fscore += max([f1_pair(pre_y, true_y)[0] for true_y in true_ys]) * cur_size
    return float(tot_fscore) / tot_size


def jc_pair(pred_y, true_y):
    """calculate jc score for a pair of communities (predicted and ground truth)

    args: 
        pred_y (N * 1): binary array, 1 means the corresponding instance belongs to predicted community
        true_y (N * 1): binary array, 1 means the corresponding instance belongs to golden community
    """
    corrected = sum([ 2 == (py + ty) for (py, ty) in zip(pred_y, true_y)])
    if 0 == corrected:
        return 0
    tot = sum([ (py + ty) > 0 for (py, ty) in zip(pred_y, true_y)])
    return float(corrected) / tot

def jc_community(pre_ys, true_ys):
    """calculate jc score for two sets of communities (predicted and ground truth)

    args: 
        pred_ys (k * N): 
        true_ys (l * N):
    """
    tot_jcscore = 0.0

    tmp_size = float(1) / ( len(pre_ys) * 2 )
    for pre_y in pre_ys:
        tot_jcscore += max([jc_pair(pre_y, true_y) for true_y in true_ys]) * tmp_size

    tmp_size = float(1) / ( len(true_ys) * 2 )
    for true_y in true_ys:
        tot_jcscore += max([jc_pair(pre_y, true_y) for pre_y in pre_ys]) * tmp_size

    return tot_jcscore

if __name__ == "__main__":
    if len(sys.argv) == 3:
        with open(sys.argv[1], 'rb') as truthfile:
            truthreader = csv.reader(truthfile)
            truth = []
            for line in truthreader:
                row = []
                for element in line:
                    row.append(int(element))
                truth.append(row)

        with open(sys.argv[2], 'rb') as predfile:
            predreader = csv.reader(predfile)
            pred = []
            for line in predreader:
                row = []
                for element in line:
                    row.append(int(element))
                pred.append(row)

        print('f1 score:')
        print(f1_community(pred, truth))
        print('jc score:')
        print(jc_community(pred, truth))
        print('nmi score:')
        print(nmi_community(pred, truth))
    else:
        y = [[1, 1, 0, 1, 0], [0, 1, 0, 1, 1], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]]
        x1 = [[0, 1, 0, 1, 1], [1, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]] #same
        x2 = [[1, 1, 1, 1, 0], [0, 1, 0, 1, 1], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]] #1 error
        x3 = [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]] #lots of error

        print('f1 score:')
        print(f1_community(x1, y))
        print(f1_community(x2, y))
        print(f1_community(x3, y))

        print('jc score:')
        print(jc_community(x1, y))
        print(jc_community(x2, y))
        print(jc_community(x3, y))

        print('nmi score:')
        print(nmi_community(x1, y))
        print(nmi_community(x2, y))
        print(nmi_community(x3, y))
