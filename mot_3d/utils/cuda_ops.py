import numpy as np
import torch
import mmdet3d
assert mmdet3d.__version__ < '1.0.0'

from mmdet3d.core import LiDARInstance3DBoxes, bbox_overlaps_3d


def calculate_iou_3d_gpu(preds, gts, mode='iou', translation=True):


    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).cuda().float()
    if isinstance(gts, np.ndarray):
        gts = torch.from_numpy(gts).cuda().float()
    
    assert isinstance(preds, torch.Tensor)

    if translation:
        trans = preds[[0], :3]
        gts[:, :3] -= trans
        preds[:, :3] -= trans

    assert preds.shape[0] > 0 and preds.shape[-1] == 7
    assert gts.shape[0] > 0 and gts.shape[-1] == 7

    preds = LiDARInstance3DBoxes(preds, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))
    gts = LiDARInstance3DBoxes(gts, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))

    ious = bbox_overlaps_3d(preds.tensor, gts.tensor, mode=mode, coordinate='lidar') #[num_preds, num_gts]
    assert ious.size(0) == preds.tensor.size(0)
    r = ious.cpu().numpy()
    return r

def calculate_iou_3d_gpu_aligned(preds, gts, mode='iou', translation=True):


    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).cuda().float()
    if isinstance(gts, np.ndarray):
        gts = torch.from_numpy(gts).cuda().float()
    
    assert isinstance(preds, torch.Tensor)

    if translation:
        trans = preds[[0], :3]
        gts[:, :3] -= trans
        preds[:, :3] -= trans

    assert preds.shape[0] > 0 and preds.shape[-1] == 7
    assert gts.shape[0] > 0 and gts.shape[-1] == 7

    preds = LiDARInstance3DBoxes(preds, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))
    gts = LiDARInstance3DBoxes(gts, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))

    ious = LiDARInstance3DBoxes.aligned_iou_3d(preds, gts) #[num_preds, num_gts]
    r = ious.cpu().numpy()
    return r

def calculate_iou_bev_gpu(preds, gts, mode='iou', translation=True):


    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).cuda().float()
    if isinstance(gts, np.ndarray):
        gts = torch.from_numpy(gts).cuda().float()
    
    if translation:
        trans = preds[[0], :3]
        gts[:, :3] -= trans
        preds[:, :3] -= trans
    
    assert isinstance(preds, torch.Tensor)

    assert preds.shape[0] > 0 and preds.shape[-1] == 7
    assert gts.shape[0] > 0 and gts.shape[-1] == 7

    preds = LiDARInstance3DBoxes(preds, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))
    gts = LiDARInstance3DBoxes(gts, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))

    ious = LiDARInstance3DBoxes.overlaps_bev(preds, gts, mode=mode) #[num_preds, num_gts]
    assert ious.size(0) == preds.tensor.size(0)
    r = ious.cpu().numpy()
    return r