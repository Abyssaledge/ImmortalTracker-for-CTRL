import os, argparse
import numpy as np
import pickle as pkl, json
from tqdm import tqdm

from waymo_open_dataset.protos import metrics_pb2

from ipdb import set_trace
import sys
sys.path.append('.')
from mot_3d.tracklet import SimpleTracklet

def generate_tracklets(bin_data, is_gt=False, cats=[1,]):
    tracklets = {}
    objects = bin_data.objects
    for o in tqdm(objects):
        obj_id = o.object.id
        cat = o.object.type

        if cat not in cats:
            continue

        if is_gt:
            obj_id = str(cat) + '_' + obj_id[:3].replace('_', 'a').replace('-', 'b')

        ts = o.frame_timestamp_micros
        seg_name = o.context_name
        obj_uuid = seg_name + '-' + obj_id
        box = o.object.box
        box_np = np.array([box.center_x, box.center_y, box.center_z, box.width, box.length, box.height, box.heading, o.score], dtype=np.float32)
        if obj_uuid not in tracklets:
            new_tracklet = SimpleTracklet(obj_uuid, cat, is_gt=is_gt)
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

def load_ego(segment_name, data_folder):
    ts_info = json.load(open(os.path.join(data_folder, 'ts_info', '{:}.json'.format(segment_name)), 'r'))
    max_frame = len(ts_info)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_info = [ego_info[str(i)] for i in range(max_frame)]
    return ego_info, ts_info

def read_bin(file_path):
    with open(file_path, 'rb') as f:
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
    return objects

parser = argparse.ArgumentParser()
parser.add_argument('bin_path', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    bin_path = args.bin_path
    bin_data = read_bin(bin_path)
    objects = bin_data.objects
    veh_cnt = 0
    ped_cnt = 0
    cyc_cnt = 0
    for o in objects:
        cat = o.object.type
        if cat == 1:
            veh_cnt += 1
        if cat == 2:
            ped_cnt += 1
        if cat == 4:
            cyc_cnt += 1
    print(f'Size of {bin_path} : {veh_cnt} vehicles, {ped_cnt} pedestrians, {cyc_cnt} cyclists')