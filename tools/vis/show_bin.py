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
parser.add_argument('--gt-bin-path', type=str, default='./gt.bin')
parser.add_argument('--save-folder', type=str, default='')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--split', type=str, default='validation')
parser.add_argument('--no-gt', action='store_true')
parser.add_argument('--only-moving', action='store_true')
parser.add_argument('--only-static', action='store_true')
parser.add_argument('--with-gt', action='store_true')
parser.add_argument('--displacement', type=float, default=2.0)
# process
args = parser.parse_args()

def seq_visualization(pts_trks, name='', save_path='./exp.png', figsize=(40, 40), x_range=None, y_range=None):
    visualizer = Visualizer2D(name=name, figsize=figsize, x_range=x_range, y_range=y_range)
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
        save_folder = args.save_folder

    assert 'vis' in save_folder

    os.makedirs(save_folder, exist_ok=True)

    bin_data = read_bin(bin_path)
    track_seg_dict = generate_tracklets(bin_data)
    if args.with_gt:
        gt_bin_path = osp.abspath(args.gt_bin_path)
        gt_bin_data = read_bin(gt_bin_path)
        gt_track_seg_dict = generate_tracklets(gt_bin_data, True)

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

            if args.only_moving and trk.displacement < args.displacement:
                continue
            if args.only_static and trk.displacement > args.displacement:
                continue

            centerpoints = trk.transformed_centerpoints(target_inv_ego)
            pt_trks.append(centerpoints)

        gt_pt_trks = []
        if args.with_gt and seg_name in gt_track_seg_dict:
            gt_trks = gt_track_seg_dict[seg_name]
            for trk in gt_trks:
                trk.add_ego(ego_list, ts_list)
                centerpoints = trk.transformed_centerpoints(target_inv_ego)
                gt_pt_trks.append(centerpoints)
            seq_visualization(
                gt_pt_trks,
                save_path=osp.join(save_folder, seg_name + args.suffix + '-gt.png'),
                figsize=(18, 18),
                x_range=(-200, 200),
                y_range=(-200, 200),
            )

        seq_visualization(
            pt_trks,
            save_path=osp.join(save_folder, seg_name + args.suffix + '.png'),
            figsize=(18, 18),
            x_range=(-200, 200),
            y_range=(-200, 200),
        )