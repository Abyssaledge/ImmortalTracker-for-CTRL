from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from tqdm import tqdm
from ipdb import set_trace
from mot_3d.tracklet import SimpleTracklet
from mot_3d.utils import load_ego
import numpy as np
import os
from os import path as osp
import argparse, yaml

def read_bin(file_path):
    with open(file_path, 'rb') as f:
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
    return objects

def generate_tracklets(bin_data):
    tracklets = {}
    objects = bin_data.objects
    for o in tqdm(objects):
        obj_id = o.object.id
        ts = o.frame_timestamp_micros
        seg_name = o.context_name
        obj_uuid = seg_name + '-' + obj_id
        box = o.object.box
        box_np = np.array([box.center_x, box.center_y, box.center_z, box.width, box.length, box.height, box.heading, o.score], dtype=np.float32)
        cat = o.object.type
        if obj_uuid not in tracklets:
            new_tracklet = SimpleTracklet(obj_uuid, cat)
            new_tracklet.append(box_np, ts)
            tracklets[obj_uuid] = new_tracklet
        else:
            tracklets[obj_uuid].append(box_np, ts)
    
    
    tracklets_seg_dict = {}
    for uuid, trk in tracklets.items():
        trk.freeze()
        segment_name = trk.segment_name
        if segment_name not in tracklets_seg_dict:
            tracklets_seg_dict[segment_name] = []
        tracklets_seg_dict[segment_name].append(trk)

    return tracklets_seg_dict

def match_tracklets(trks1, trks2, cfg):
    cost = np.zeros((len(trks1), len(trks2)))
    for i in range(len(trks1)):
        for k in range(len(trks2)):
            dis = SimpleTracklet.tracklet_distance(trks1[i], trks2[k], cfg=cfg)
            cost[i, k] = dis
    argmin = cost.argmin(1)
    min_cost = cost.min(1)
    matched_min = min_cost < cfg['cost_thresh']
    matched_pairs = []
    for i in range(len(argmin)):
        if matched_min[i].item():

            if cfg.get('only_moving', False) and trks1[i].displacement < 2:
                continue
            if cfg.get('only_static', False) and trks1[i].displacement > 2:
                continue

            pair = (i, argmin[i].item())
            matched_pairs.append(pair)


    return matched_pairs

def merge_tracklets(trks1, trks2, pairs, cfg):

    out_trks = []

    if cfg.get('keep_unmatch1', False):
        if len(pairs) == 0:
            out_trks = trks1
        else:
            indices_1 = np.array([p[0] for p in pairs], dtype=np.int)
            unmatched_1 = np.ones(indices_1.max().item() + 1, dtype=bool)
            unmatched_1[indices_1] = False
            unmatched_1 = np.where(unmatched_1)[0].tolist()
            out_trks += [trks1[i] for i in unmatched_1]

    if cfg.get('keep_unmatch2', False):

        max_id_1 = max([t.int_id for t in trks1])
        for t2 in trks2:
            t2.increase_id(max_id_1 + 1)

        if len(pairs) == 0:
            out_trks += trks2
        else:
            indices_2 = np.array([p[1] for p in pairs], dtype=np.int)
            unmatched_2 = np.ones(indices_2.max().item() + 1, dtype=bool)
            unmatched_2[indices_2] = False
            unmatched_2 = np.where(unmatched_2)[0].tolist()
            out_trks += [trks2[i] for i in unmatched_2]
    

    for pair in pairs:
        merged_trk = SimpleTracklet.merge_tracklets([trks1[pair[0]], trks2[pair[1]]], cfg)
        out_trks.append(merged_trk)
    return out_trks

def save_to_bin(tracklet_dict, save_path):
    objects = metrics_pb2.Objects()
    for seg_name, trks in tracklet_dict.items():
        for trk in trks:
            waymo_objs = trk.waymo_objects()
            for obj in waymo_objs:
                objects.objects.append(obj)

    f = open(save_path, 'wb')
    f.write(objects.SerializeToString())
    f.close()

def check_unique_id(trks):
    num_trks = len(trks)
    ids = set([t.type_and_id for t in trks])
    assert len(ids) == num_trks


def call_bin(save_path):
    import subprocess
    print('Start evaluating bin file...')
    ret_bytes = subprocess.check_output(
        f'./compute_tracking_metrics_main_official {save_path} ' + './gt.bin',
        shell=True)
    ret_texts = ret_bytes.decode('utf-8')
    print(ret_texts)
    txt_path = save_path.replace('.bin', '.txt')
    with open(txt_path, 'w') as fw:
        fw.write(ret_texts)


parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--bin_path_1', type=str, default='./mot_results/waymo/validation/immortal_gpu_real3d_fsdpp/bin/pred.bin')
parser.add_argument('--bin_path_2', type=str, default='./mot_results/waymo/validation/immortal_backward_re_fsdpp/bin/pred.bin')
parser.add_argument('--device', type=int, default=-1)
parser.add_argument('--split', type=str, default='validation')
args = parser.parse_args()

if __name__ == '__main__':
    if args.device > -1:
        import torch
        torch.cuda.set_device(args.device)
    bin_path_1 = args.bin_path_1
    bin_path_2 = args.bin_path_2
    bin_data_1 = read_bin(bin_path_1)
    bin_data_2 = read_bin(bin_path_2)
    save_name = osp.basename(args.config).split('.')[0]
    save_path = osp.join(osp.dirname(bin_path_1), save_name + '.bin')
    print(f'Results will be saved to {save_path}')

    tracklet_dict_1 = generate_tracklets(bin_data_1)
    tracklet_dict_2 = generate_tracklets(bin_data_2)


    out_tracklet_dict = {}

    assert set(tracklet_dict_1.keys()) == set(tracklet_dict_2.keys()), 'It is almost impossible that a sequence does not contain any tracklet'

    cfg = yaml.load(open(args.config, 'r'))
    if 'test' in args.split:
        data_folder = os.path.join('./data/waymo', 'testing')
    else:
        data_folder = os.path.join('./data/waymo', 'validation')
        
    print('Start match and merge...')
    for segment_name, trks_1 in tqdm(tracklet_dict_1.items()):
        trks_2 = tracklet_dict_2[segment_name]

        if cfg.get('only_moving', False) or cfg.get('only_static', False):
            ego_list, ts_list = load_ego('segment-' + seg_name + '_with_camera_labels', data_folder)
            for trk in trks_1 + trks_2:
                trk.add_ego(ego_list, ts_list)
            set_trace()

        pairs = match_tracklets(trks_1, trks_2, cfg=cfg)
        merged_trks = merge_tracklets(trks_1, trks_2, pairs, cfg)
        check_unique_id(merged_trks)
        out_tracklet_dict[segment_name] = merged_trks
    
    save_to_bin(out_tracklet_dict, save_path)
    call_bin(save_path)