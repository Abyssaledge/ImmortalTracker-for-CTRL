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
parser.add_argument('--bin-path', type=str, default='./mot_results/waymo/validation/immortal_gpu_real3d_fsdpp/bin/pred.bin')
parser.add_argument('--split', type=str, default='validation')
args = parser.parse_args()

if __name__ == '__main__':
    bin_path = args.bin_path
    bin_data = read_bin(bin_path)
    old_bin_name = osp.basename(bin_path).split('.')[0]

    suffix = osp.basename(args.config).split('.')[0]

    save_path = osp.join(osp.dirname(bin_path), old_bin_name + f'_{suffix}.bin')
    print(f'Results will be saved to {save_path}')

    tracklet_dict = generate_tracklets(bin_data)
    cfg = yaml.load(open(args.config, 'r'))
    if 'test' in args.split:
        data_folder = os.path.join('./data/waymo', 'testing')
    else:
        data_folder = os.path.join('./data/waymo', 'validation')

    print('Start interpolating...')
    for seg_name, trks in tqdm(tracklet_dict.items()):
        ego_list, ts_list = load_ego('segment-' + seg_name + '_with_camera_labels', data_folder)
        for trk in trks:
            trk.interpolate(cfg, ts_list, ego_list)
    
    save_to_bin(tracklet_dict, save_path)
    call_bin(save_path)