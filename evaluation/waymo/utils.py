from mot_3d.tracklet import SimpleTracklet
import numpy as np
from scipy.optimize import linear_sum_assignment
from ipdb import set_trace
from collections import defaultdict
from tqdm import tqdm
import os
from os import path as osp

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

            if cfg.get('only_moving', False) and trks1[i].displacement <= moving_thresh:
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

def greedy_match(trks1, trks2, cfg):
    cost = np.zeros((len(trks1), len(trks2)))
    for i in range(len(trks1)):
        for k in range(len(trks2)):
            dis = SimpleTracklet.tracklet_distance(trks1[i], trks2[k], cfg=cfg)
            cost[i, k] = dis

    num_trks1, num_trks2 = len(trks1), len(trks2)

    # association in the greedy manner
    cost_1d = cost.reshape(-1)
    index_1d = np.argsort(cost_1d)
    index_2d = np.stack([index_1d // num_trks2, index_1d % num_trks2], axis=1)
    trks1_to_trks2_id = [-1] * num_trks1
    trks2_to_trks1_id = [-1] * num_trks2
    matched_pairs = []
    for sort_i in range(index_2d.shape[0]):
        id1 = int(index_2d[sort_i][0])
        id2 = int(index_2d[sort_i][1])
        if trks2_to_trks1_id[id2] == -1 and trks1_to_trks2_id[id1] == -1:
            trks2_to_trks1_id[id2] = id1
            trks1_to_trks2_id[id1] = id2
            matched_pairs.append((id1, id2))

    return matched_pairs

def bi_naive_match(trks1, trks2, cfg):
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

    cost = cost.T
    argmin = cost.argmin(1)
    min_cost = cost.min(1)
    matched_min = min_cost < cfg['cost_thresh']
    for i in range(len(argmin)):
        if matched_min[i].item():

            if cfg.get('only_moving', False) and trks2[i].displacement < moving_thresh:
                continue
            if cfg.get('only_static', False) and trks2[i].displacement > moving_thresh:
                continue

            pair = (argmin[i].item(), i)
            matched_pairs.append(pair)
    matched_pairs = list(set(matched_pairs))

    return matched_pairs

def mutual_naive_match(trks1, trks2, cfg):
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

    cost = cost.T
    argmin = cost.argmin(1)
    min_cost = cost.min(1)
    matched_min = min_cost < cfg['cost_thresh']
    matched_pairs2 = []
    for i in range(len(argmin)):
        if matched_min[i].item():

            if cfg.get('only_moving', False) and trks2[i].displacement < moving_thresh:
                continue
            if cfg.get('only_static', False) and trks2[i].displacement > moving_thresh:
                continue

            pair = (argmin[i].item(), i)
            matched_pairs2.append(pair)
    mutual_matched_pairs = list(set(matched_pairs).intersection(set(matched_pairs2)))

    return mutual_matched_pairs


def waymo_object_to_mmdet(obj, version):
    '''
    According to https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto#L33
    and the definition of LiDARInstance3DBoxes
    '''
    # if gt and obj.object.num_lidar_points_in_box == 0:
    #     print('Encounter zero-point object')
    #     return None
    box = obj.object.box

    assert version < '1.0.0', 'Only support version older than 1.0.0 for now'
    heading = -box.heading - 0.5 * np.pi

    while heading < -np.pi:
        heading += 2 * np.pi
    while heading > np.pi:
        heading -= 2 * np.pi

    result = np.array(
        [
            box.center_x,
            box.center_y,
            box.center_z,
            box.width,
            box.length,
            box.height,
            heading,
            # obj.score,
            # float(obj.object.type),
        ]
    )
    return result

def waymo_object_to_array(obj):
    box = obj.object.box
    result = np.array(
        [
            box.center_x,
            box.center_y,
            box.center_z,
            box.width,
            box.length,
            box.height,
            box.heading,
            obj.score,
            float(obj.object.type),
        ]
    )
    return result

def bin2lidarboxes(bin_data, debug=False, gt=False):
    # import mmdet3d
    # from mmdet3d.core import LiDARInstance3DBoxes
    # mmdet3d_version = mmdet3d.__version__
    objects = bin_data.objects
    obj_dict = defaultdict(list)
    ori_obj_dict = defaultdict(list)
    id_dict = defaultdict(list)
    segname_dict = {}
    print('Collecting Bboxes ...')
    for o in tqdm(objects):
        seg_name = o.context_name
        time_stamp = o.frame_timestamp_micros
        obj_id = o.object.id
        mm_obj = waymo_object_to_mmdet(o, '0.15.0')
        if mm_obj is not None:
            obj_dict[time_stamp].append(mm_obj)
            ori_obj_dict[time_stamp].append(waymo_object_to_array(o))
            id_dict[time_stamp].append(obj_id)
            segname_dict[time_stamp] = seg_name

    out_list = []

    for ts in tqdm(obj_dict):

        boxes = np.stack(obj_dict[ts], axis=0)[:, :7]
        # boxes = LiDARInstance3DBoxes(np.stack(boxes, axis=0)[:, :7], box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))
        ori_boxes = np.stack(ori_obj_dict[ts])
        ids = np.stack(id_dict[ts])

        e = (ori_boxes, boxes, ids, segname_dict[ts], ts)
        out_list.append(e)

    return out_list


def get_pc_from_time_stamp(timestamp, ts2idx, data_root, split='training'):

    curr_idx = ts2idx[timestamp]
    pc_root = osp.join(data_root, f'{split}/velodyne')
    pc_path = os.path.join(pc_root, curr_idx + '.bin')
    pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 6)
    pc = pc[:, :3]
    return pc