import numpy as np
from ipdb import set_trace

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
        self.segment_name = uuid.split('-')[0]
        self.type = type
        self.size = len(self.box_list)

    
    def append(self, box, ts, id=None):
        self.box_list.append(box)
        self.ts_list.append(ts)
        self.size += 1
        if id is not None:
            assert self.id == id
    
    def freeze(self):
        self.ts2index = {ts:i for i, ts in enumerate(self.ts_list)}
    
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
    def overlap_ts(cls, trk1, trk2):
        s1 = set(trk1.ts_list)
        s2 = set(trk2.ts_list)
        inter = s1.intersection(s2)
        # union = s1.union(s2)
        unq1 = sorted(list(s1 - s2))
        unq2 = sorted(list(s2 - s1))
        overlap_list = sorted(list(inter))
        return overlap_list, unq1, unq2
    
    @classmethod
    def tracklet_distance(cls, trk1, trk2, mode='center_distance', cfg=None):
        return getattr(cls, mode)(trk1, trk2, cfg)
    
    @classmethod
    def center_distance(cls, trk1, trk2, cfg):
        overlap_ts, unq_ts1, unq_ts2 = cls.overlap_ts(trk1, trk2)
        num_unq = len(unq_ts1) + len(unq_ts2)
        if len(overlap_ts) == 0:
            return cfg['max_distance']
        boxes_1 = np.stack([trk1[ts] for ts in overlap_ts], 0)
        boxes_2 = np.stack([trk2[ts] for ts in overlap_ts], 0)
        center_dist = np.linalg.norm(boxes_1[:, :3] - boxes_2[:, :3], ord=2, axis=1)
        avg_dist = (center_dist.sum().item() + num_unq * cfg['max_distance']) / (num_unq + len(overlap_ts))
        return avg_dist
