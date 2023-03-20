from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from tqdm import tqdm
from ipdb import set_trace
import sys
sys.path.append('.')
from mot_3d.tracklet import SimpleTracklet
from mot_3d.utils import load_ego
import numpy as np
import os
from os import path as osp
import argparse, yaml
from utils import naive_match, hungarian_match, greedy_match, bi_naive_match, mutual_naive_match

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
    mode = cfg.get('match', 'naive')
    if mode == 'naive':
        return naive_match(trks1, trks2, cfg)
    if mode == 'hungarian':
        return hungarian_match(trks1, trks2, cfg)
    if mode == 'greedy':
        return greedy_match(trks1, trks2, cfg)
    if mode == 'bi_naive':
        return bi_naive_match(trks1, trks2, cfg)
    if mode == 'mutual_naive':
        return mutual_naive_match(trks1, trks2, cfg)
    raise NotImplementedError 
matched_cnt = 0 
unmatched_cnt = 0
def merge_tracklets(trks1, trks2, pairs, cfg):

    out_trks = []
    global matched_cnt
    global unmatched_cnt
    matched_cnt += len(pairs)

    unmatched_cnt += len(trks1) - len(pairs)
    if len(pairs) == 0:
        unmatched_trks_1 = trks1
    else:
        indices_1 = np.array([p[0] for p in pairs], dtype=np.int)
        unmatched_1 = np.ones(len(trks1), dtype=bool)
        unmatched_1[indices_1] = False
        unmatched_1 = np.where(unmatched_1)[0].tolist()
        unmatched_trks_1 = [trks1[i] for i in unmatched_1]

    if cfg.get('keep_unmatch1', False):
        out_trks += unmatched_trks_1

    unmatched_cnt += len(trks2) - len(pairs)

    max_id_1 = max([t.int_id for t in trks1])
    for t2 in trks2:
        t2.increase_id(max_id_1 + 1)

    if len(pairs) == 0:
        unmatched_trks_2 = trks2
    else:
        indices_2 = np.array([p[1] for p in pairs], dtype=np.int)
        unmatched_2 = np.ones(len(trks2), dtype=bool)
        unmatched_2[indices_2] = False
        unmatched_2 = np.where(unmatched_2)[0].tolist()
        unmatched_trks_2 = [trks2[i] for i in unmatched_2]

    if cfg.get('keep_unmatch2', False):
        out_trks += unmatched_trks_2
    

    for pair in pairs:
        merged_trk = SimpleTracklet.merge_tracklets([trks1[pair[0]], trks2[pair[1]]], cfg)
        out_trks.append(merged_trk)
    return out_trks, unmatched_trks_1, unmatched_trks_2

def save_to_bin(tracklet_dict, save_path, prefix=None):
    objects = metrics_pb2.Objects()
    for seg_name, trks in tracklet_dict.items():
        for trk in trks:
            waymo_objs = trk.waymo_objects()
            for obj in waymo_objs:
                objects.objects.append(obj)

    if prefix is not None:
        dir_path = osp.dirname(save_path)
        save_path = osp.join(dir_path, prefix + '_' + osp.basename(save_path))
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
        f'./compute_tracking_metrics_main_detail {save_path} ' + './gt.bin',
        shell=True)
    ret_texts = ret_bytes.decode('utf-8')
    print(ret_texts)
    txt_path = save_path.replace('.bin', '.txt')
    with open(txt_path, 'w') as fw:
        fw.write(ret_texts)


parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--bin_path_1', type=str, default='./mot_results/waymo/validation/tl_basic_maxtime10_fsd_pastfuture/bin/vehicle/pred.bin')
parser.add_argument('--bin_path_2', type=str, default='./mot_results/waymo/validation/tl_basic_maxtime10_back_fsd_pastfuture/bin/vehicle/pred.bin')
parser.add_argument('--device', type=int, default=0)
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
    unmatched_tracklet_dict1 = {}
    unmatched_tracklet_dict2 = {}

    assert set(tracklet_dict_1.keys()) == set(tracklet_dict_2.keys()), 'It is almost impossible that a sequence does not contain any tracklet'

    cfg = yaml.load(open(args.config, 'r'))
    if 'test' in args.split:
        data_folder = os.path.join('./data/waymo', 'testing')
    else:
        data_folder = os.path.join('./data/waymo', 'validation')
    
    if cfg.get('swap', False):
        tracklet_dict_1, tracklet_dict_2 = tracklet_dict_2, tracklet_dict_1
        print('Swap two results.')
        
    print('Start match and merge...')
    for segment_name, trks_1 in tqdm(tracklet_dict_1.items()):
        trks_2 = tracklet_dict_2[segment_name]

        pairs = match_tracklets(trks_1, trks_2, cfg=cfg)
        merged_trks, unmatched_trks1, unmatched_trks2 = merge_tracklets(trks_1, trks_2, pairs, cfg)
        check_unique_id(merged_trks)
        out_tracklet_dict[segment_name] = merged_trks
        unmatched_tracklet_dict1[segment_name] = unmatched_trks1
        unmatched_tracklet_dict2[segment_name] = unmatched_trks2
    
    print(f'matched_cnt:{matched_cnt}, unmatched_cnt:{unmatched_cnt}')
    length = 0
    cnt = 0
    for _, trks in unmatched_tracklet_dict1.items():
        length += sum([len(t) for t in trks])
        cnt += len(trks)
    for _, trks in unmatched_tracklet_dict2.items():
        length += sum([len(t) for t in trks])
        cnt += len(trks)

    print('mean length of unmatches: ', length / cnt)
    save_to_bin(out_tracklet_dict, save_path)
    # save_to_bin(unmatched_tracklet_dict1, save_path, prefix='unmatch1')
    # save_to_bin(unmatched_tracklet_dict2, save_path, prefix='unmatch2')
    call_bin(save_path)