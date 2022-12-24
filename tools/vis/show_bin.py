import numpy as np
import os, sys
from os import path as osp
import argparse
from ipdb import set_trace
from tqdm import tqdm
sys.path.append('.')

# from pipeline_vis import frame_visualization
from visualizer import Visualizer2D
from utils import generate_tracklets, read_bin, load_ego

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--bin-path', type=str, default='./mot_results/waymo/validation/immortal_gpu_real3d_fsdpp/bin/pred.bin')
parser.add_argument('--gt-bin-path', type=str, default='./data/waymo/waymo_format/gt.bin')
parser.add_argument('--save-folder', type=str, default='./work_dirs/vis_folder')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--split', type=str, default='validation')
parser.add_argument('--no-gt', action='store_true')
# process
args = parser.parse_args()

def seq_visualization(pts_trks, name='', save_path='./exp.png', figsize=(40, 40)):
    visualizer = Visualizer2D(name=name, figsize=figsize)
    for i, p in enumerate(pts_trks):
        visualizer.handler_tracklet(p, id=i, color=i, s=0.2)
    visualizer.save(save_path)
    print(f'Saved to {save_path}')
    visualizer.close()

if __name__ == '__main__':
    bin_path = osp.abspath(args.bin_path)
    bin_name = osp.basename(bin_path).split('.')[0]
    if args.save_folder == '':
        save_folder = osp.join('./work_dirs/vis_folder/', bin_name)
    else: 
        assert args.suffix != ''
        save_folder = args.save_folder

    assert 'vis' in save_folder

    os.makedirs(save_folder, exist_ok=True)

    bin_data = read_bin(bin_path)
    track_seg_dict = generate_tracklets(bin_data)

    if 'test' in args.split:
        data_folder = os.path.join('./data/waymo', 'testing')
    else:
        data_folder = os.path.join('./data/waymo', 'validation')

    for seg_name, trks in track_seg_dict.items():
        ego_list, ts_list = load_ego('segment-' + seg_name + '_with_camera_labels', data_folder)
        target_inv_ego = np.linalg.inv(ego_list[0])
        pt_trks = []
        for trk in trks:
            trk.add_ego(ego_list, ts_list)
            centerpoints = trk.transformed_centerpoints(target_inv_ego)
            pt_trks.append(centerpoints)

        seq_visualization(
            pt_trks,
            save_path=osp.join(save_folder, seg_name + args.suffix + '.png'),
            figsize=(18, 18)
        )