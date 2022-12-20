from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from tqdm import tqdm
from ipdb import set_trace
from mot_3d.tracklet import SimpleTracklet
import numpy as np

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
        box_np = np.array([box.center_x, box.center_y, box.center_z, box.width, box.length, box.height, box.heading], dtype=np.float32)
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

def match_tracklets(trks1, trks2, mode, cfg):
    cost = np.zeros((len(trks1), len(trks2)))
    for i in range(len(trks1)):
        for k in range(len(trks2)):
            dis = SimpleTracklet.tracklet_distance(trks1[i], trks2[k], mode=mode, cfg=cfg)
            cost[i, k] = dis
    set_trace()


if __name__ == '__main__':
    bin_path_1 = './mot_results/waymo/validation/immortal_gpu_real3d_fsdpp/bin/pred.bin'
    bin_path_2 = './mot_results/waymo/validation/immortal_backward_re_fsdpp/bin/pred.bin'
    bin_data_1 = read_bin(bin_path_1)
    bin_data_2 = read_bin(bin_path_2)

    tracklet_dict_1 = generate_tracklets(bin_data_1)
    tracklet_dict_2 = generate_tracklets(bin_data_2)

    assert set(tracklet_dict_1.keys()) == set(tracklet_dict_2.keys()), 'It is almost impossible that a sequence does not contain any tracklet'

    cfg = {'max_distance':10}
    for segment_name, trks_1 in tracklet_dict_1.items():
        trks_2 = tracklet_dict_2[segment_name]
        indices = match_tracklets(trks_1, trks_2, mode='center_distance', cfg=cfg)