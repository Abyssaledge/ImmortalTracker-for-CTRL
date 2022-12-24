import numpy as np
from ipdb import set_trace
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

class SimpleTracklet(object):

    def __init__(self, uuid, type, box_list=None, ts_list=None):

        if box_list is None:
            self.box_list = []
            self.ts_list = []
        else:
            self.box_list = box_list
            self.ts_list = ts_list

        self.uuid = uuid
        assert '-' in uuid
        self.segment_name, self.type_and_id = uuid.split('-')
        self.type = type
        checktype, int_id = self.type_and_id.split('_')
        self.int_id = int(int_id)
        assert isinstance(self.type, int)
        assert int(checktype) == self.type
        self.size = len(self.box_list)
        self.frozen = False

    
    def append(self, box, ts, uuid=None):
        self.box_list.append(box)
        self.ts_list.append(ts)
        self.size += 1
        if uuid is not None:
            assert self.uuid == uuid
    
    @staticmethod
    def points_frame_transform(pre_points, pre_pose, cur_pose_inv):
        pre_points_h = np.pad(pre_points, (0, 1), 'constant', constant_values=1)

        world2curr_pose = cur_pose_inv

        mm = world2curr_pose @ pre_pose
        pre_points_in_cur = (pre_points_h @ mm.T)[:3]
        return pre_points_in_cur
    
    @property
    def centerpoints(self,):
        return [b[:3] for b in self.box_list]

    def transformed_centerpoints(self, target_inv_ego):
        pts = self.centerpoints
        assert len(pts) == len(self.ego_list)
        out_pts = np.stack([self.points_frame_transform(p, ego, target_inv_ego) for p, ego in zip(pts, self.ego_list)], 0)
        return out_pts

    
    def add_ego(self, ego_list, ts_list):
        tmp = {t:e for t, e in zip(ts_list, ego_list)}
        self.ego_list = [tmp[ts] for ts in self.ts_list]
        self.inv_ego_list = [np.linalg.inv(e) for e in self.ego_list]
    
    def freeze(self):
        self.ts2index = {ts:i for i, ts in enumerate(self.ts_list)}
        assert self.ts_list == sorted(self.ts_list)
        self.frozen = True
    
    @property
    def displacement(self,):
        # a stupid way when there is no availble velocity
        assert self.frozen
        if hasattr(self, '_displacement'):
            return self._displacement
        else:
            assert hasattr(self, 'ego_list')
            ego = self.ego_list[0]
            target_inv_ego = self.inv_ego_list[-1]
            c1 = self.points_frame_transform(self.box_list[0][:3], ego, target_inv_ego)
            self._displacement = np.linalg.norm(self.box_list[-1][:3] - c1, ord=2)
            return self._displacement

    
    def __getitem__(self, key):
        assert isinstance(key, int)
        if key > 1e10:
            # shoud be a timestamp
            return self.box_list[self.ts2index[key]]
        elif key < self.size:
            return self.box_list[key]
        else:
            raise KeyError
    
    def __len__(self):
        return self.size
    
    @classmethod
    def boxes2mm(cls, boxes):
        boxes = boxes[:, :7].copy()
        heading = -boxes[:, 6] - 0.5 * np.pi
        
        while (heading < -np.pi).any():
            heading[heading < -np.pi] += 2 * np.pi

        while (heading > np.pi).any():
            heading[heading > np.pi] -= 2 * np.pi

        boxes[:, 6] = heading

        return boxes
    
    @classmethod
    def overlap_ts(cls, trk1, trk2):
        s1 = set(trk1.ts_list)
        s2 = set(trk2.ts_list)
        inter = s1.intersection(s2)
        # union = s1.union(s2)
        return inter, s1 - s2, s2 - s1
    
    def ts_iou(self, trk_b):
        sa = set(self.ts_list)
        sb = set(trk_b.ts_list)
        inter = len(sa.intersection(sb))
        union = len(sa.union(sb))
        assert union > 0
        return inter / union

    def ts_iox(self, trk_b):
        sa = set(self.ts_list)
        sb = set(trk_b.ts_list)
        inter = len(sa.intersection(sb))
        assert len(sb) > 0
        return inter / len(sb)

    def ts_ios(self, trk_b): # intersection over small 
        sa = set(self.ts_list)
        sb = set(trk_b.ts_list)
        inter = len(sa.intersection(sb))
        small = min(len(sa), len(sb))
        assert small > 0
        return inter / small
    
    
    @classmethod
    def tracklet_distance(cls, trk1, trk2, cfg):
        return getattr(cls, cfg['mode'])(trk1, trk2, cfg)
    
    @classmethod
    def center_distance(cls, trk1, trk2, cfg):
        overlap_ts, unq_ts1, unq_ts2 = cls.overlap_ts(trk1, trk2)
        overlap_ts = sorted(list(overlap_ts))
        unq_ts1 = sorted(list(unq_ts1))
        unq_ts2 = sorted(list(unq_ts2))
        num_unq = len(unq_ts1) + len(unq_ts2)
        min_overlap = cfg.get('min_overlap', 1)
        if len(overlap_ts) < min_overlap:
            return cfg['max_distance']
        boxes_1 = np.stack([trk1[ts] for ts in overlap_ts], 0)
        boxes_2 = np.stack([trk2[ts] for ts in overlap_ts], 0)
        if cfg.get('with_dimensions', False):
            center_dist = np.linalg.norm(boxes_1[:, :6] - boxes_2[:, :6], ord=2, axis=1)
        else:
            center_dist = np.linalg.norm(boxes_1[:, :3] - boxes_2[:, :3], ord=2, axis=1)
        # avg_dist = (center_dist.sum().item() + num_unq * cfg['max_distance']) / (num_unq + len(overlap_ts))
        avg_dist = center_dist.mean()
        return avg_dist

    @classmethod
    def iou3d(cls, trk1, trk2, cfg):
        from mot_3d.utils.cuda_ops import calculate_iou_3d_gpu
        overlap_ts, unq_ts1, unq_ts2 = cls.overlap_ts(trk1, trk2)
        overlap_ts = sorted(list(overlap_ts))
        unq_ts1 = sorted(list(unq_ts1))
        unq_ts2 = sorted(list(unq_ts2))
        num_unq = len(unq_ts1) + len(unq_ts2)
        min_overlap = cfg.get('min_overlap', 1)
        if len(overlap_ts) < min_overlap:
            return cfg['max_distance']
        boxes_1 = np.stack([trk1[ts] for ts in overlap_ts], 0)
        boxes_2 = np.stack([trk2[ts] for ts in overlap_ts], 0)
        center_dist = np.linalg.norm(boxes_1[:, :3] - boxes_2[:, :3], ord=2, axis=1).mean().item()

        if center_dist > 20:
            ious = np.zeros(len(boxes_1))
        else:
            boxes_1 = cls.boxes2mm(boxes_1)
            boxes_2 = cls.boxes2mm(boxes_2)
            iou_matrix = calculate_iou_3d_gpu(boxes_1, boxes_2)
            ious = np.diagonal(iou_matrix)
        
        if cfg.get('timestamp_ios', False):
            ios = trk1.ts_ios(trk2)
            ious = ious * ios

        if cfg.get('tracklet_ios', False):
            trk_ios = (ious > cfg['box_matched_thresh']).sum() / min(len(trk1), len(trk2))
            ious = ious * trk_ios

        if cfg.get('tracklet_iou', False):
            trk_inter = (ious > cfg['box_matched_thresh']).sum()
            trk_iou = trk_inter / (len(trk1) + len(trk2) - trk_inter)
            ious = ious * trk_iou


        dist = 1 - ious
        avg_dist = dist.mean()
        return avg_dist
    
    @classmethod
    def weighted_box_fuse(cls, box1, box2, cfg):
        s1, s2 = box1[7], box2[7]
        new_box = (box1 * s1 + box2 * s2) / (s1 + s2)
        return new_box

    @classmethod
    def max_box_fuse(cls, box1, box2, cfg):
        s1, s2 = box1[7].item(), box2[7].item()
        if s1 > s2:
            return box1
        else:
            return box2

    @classmethod
    def first_priority_fuse(cls, box1, box2, cfg):
        return box1

    @classmethod
    def merge_tracklets(cls, trks, cfg):
        # two trks for now
        trk1, trk2 = trks[0], trks[1]
        overlap_ts, unq_ts1, unq_ts2 = cls.overlap_ts(trk1, trk2)
        all_ts = sorted(list(overlap_ts.union(unq_ts1).union(unq_ts2)))
        out_trk = SimpleTracklet(trk1.uuid, trk1.type)
        for ts in all_ts:
            if ts in overlap_ts:
                box1, box2 = trk1[ts], trk2[ts]
                box_fuse_func = cfg.get('box_fuse_func', None)
                if box_fuse_func is not None:
                    new_box = getattr(cls, box_fuse_func)(box1, box2, cfg)
                else:
                    new_box = (box1 + box2) / 2
                out_trk.append(new_box, ts)
                pass
            elif ts in unq_ts1:
                box1 = trk1[ts]
                out_trk.append(box1, ts)
            else:
                box2 = trk2[ts]
                out_trk.append(box2, ts)
        return out_trk
    
    def waymo_objects(self,):
        out_list = []
        for box_np, ts in zip(self.box_list, self.ts_list):
            o = metrics_pb2.Object()
            o.context_name = self.segment_name
            o.frame_timestamp_micros = ts
            box = label_pb2.Label.Box()

            box.center_x, box.center_y, box.center_z, box.width, box.length, box.height, box.heading, o.score = box_np.tolist()
            o.object.box.CopyFrom(box)

            # meta_data = label_pb2.Label.Metadata()
            # if args.velo:
            #     meta_data.speed_x, meta_data.speed_y = velo[0], velo[1]
            # if args.accel:
            #     meta_data.accel_x, meta_data.accel_y = accel[0], accel[1]
            # o.object.metadata.CopyFrom(meta_data)

            o.object.id = self.type_and_id
            o.object.type = self.type
            out_list.append(o)
        return out_list
    
    def increase_id(self, base):
        new_id = self.int_id + base
        self.int_id = new_id
        self.type_and_id = str(self.type) + '_' + str(new_id)
        self.uuid = self.segment_name + '-' + self.type_and_id
        