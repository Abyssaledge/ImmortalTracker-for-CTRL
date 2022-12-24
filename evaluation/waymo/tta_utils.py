from mot_3d.tracklet import SimpleTracklet
import numpy as np
from scipy.optimize import linear_sum_assignment
from ipdb import set_trace

def naive_match(trks1, trks2, cfg):
    cost = np.zeros((len(trks1), len(trks2)))
    for i in range(len(trks1)):
        for k in range(len(trks2)):
            dis = SimpleTracklet.tracklet_distance(trks1[i], trks2[k], cfg=cfg)
            cost[i, k] = dis
    argmin = cost.argmin(1)
    min_cost = cost.min(1)
    matched_min = min_cost < cfg['cost_thresh']
    matched_pairs = []
    moving_thresh = cfg.get('moving_thresh', 2)
    for i in range(len(argmin)):
        if matched_min[i].item():

            if cfg.get('only_moving', False) and trks1[i].displacement < moving_thresh:
                continue
            if cfg.get('only_static', False) and trks1[i].displacement > moving_thresh:
                continue

            pair = (i, argmin[i].item())
            matched_pairs.append(pair)

    return matched_pairs

def hungarian_match(trks1, trks2, cfg):
    cost = np.zeros((len(trks1), len(trks2)))
    for i in range(len(trks1)):
        for k in range(len(trks2)):
            dis = SimpleTracklet.tracklet_distance(trks1[i], trks2[k], cfg=cfg)
            cost[i, k] = dis
    row_ind, col_ind = linear_sum_assignment(cost)
    row_ind = row_ind.tolist()
    col_ind = col_ind.tolist()

    assert len(row_ind) == len(col_ind)
    cost_thr = cfg['cost_thresh']

    matched_pairs = []
    for i in range(len(row_ind)):
        r, c = row_ind[i], col_ind[i]
        this_dist = cost[r, c]
        if this_dist < cost_thr:
            matched_pairs.append((r, c))
    return matched_pairs