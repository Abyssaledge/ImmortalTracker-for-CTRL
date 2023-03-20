""" Example of data loader:
    The data loader has to be an iterator:
    Return a dict of frame data
    Users may create the logic of your own data loader
"""
import os, numpy as np, json
import mot_3d.utils as utils
from mot_3d.data_protos import BBox
from mot_3d.preprocessing import nms
timer = utils.Timer(10)
from ipdb import set_trace
import time
import copy


class WaymoLoader:
    def __init__(self, configs, type_token, segment_name, data_folder, det_data_folder, start_frame):
        """ initialize with the path to data 
        Args:
            data_folder (str): root path to your data
        """
        self.configs = configs
        self.segment = segment_name
        self.data_loader = data_folder
        self.det_data_folder = det_data_folder
        self.type_token = type_token

        self.nms = configs['data_loader']['nms']
        self.nms_thres = configs['data_loader']['nms_thres']

        self.ts_info = json.load(open(os.path.join(data_folder, 'ts_info', '{:}.json'.format(segment_name)), 'r'))
        self.ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
            allow_pickle=True)
        self.dets = np.load(os.path.join(det_data_folder, 'dets', '{:}.npz'.format(segment_name)), allow_pickle=True)
        self.det_type_filter = True

        self.max_frame = len(self.dets['bboxes'])
        self.abs_frame_indices = list(range(self.max_frame))
        self.cur_frame = start_frame

        self.dets_bboxes = [self.dets['bboxes'][i] for i in range(self.max_frame)]
        self.dets_types = [self.dets['types'][i] for i in range(self.max_frame)]
        self.ego_info = [self.ego_info[str(i)] for i in range(self.max_frame)]
        self.det_keys = self.dets.files
        self.dets = None
        if configs['data_loader'].get('backward', True):
            # assert 'velos' not in self.dets.keys()
            self.reverse()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_frame >= self.max_frame:
            raise StopIteration

        result = dict()
        result['time_stamp_int'] = self.ts_info[self.cur_frame]
        result['time_stamp'] = self.ts_info[self.cur_frame] * 1e-6
        if isinstance(self.ego_info, list):
            result['ego'] = self.ego_info[self.cur_frame]
        else:
            result['ego'] = self.ego_info[str(self.cur_frame)]

        # bboxes = self.dets['bboxes'][self.cur_frame]
        # inst_types = self.dets['types'][self.cur_frame]
        bboxes = self.dets_bboxes[self.cur_frame]
        inst_types = self.dets_types[self.cur_frame]

        selected_dets = [bboxes[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['det_types'] = [inst_types[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['dets'] = [BBox.bbox2world(result['ego'], BBox.array2bbox(b))
            for b in selected_dets]

        result['pc'] = None
        result['aux_info'] = {'is_key_frame': True}
        
        if 'velos' in self.det_keys:
            cur_frame_velos = self.dets['velos'][self.cur_frame]
            result['aux_info']['velos'] = [np.array(cur_frame_velos[i]) 
                for i in range(len(bboxes)) if inst_types[i] in self.type_token]
            result['aux_info']['velos'] = [utils.velo2world(result['ego'], v) 
                for v in result['aux_info']['velos']]
        else:
            result['aux_info']['velos'] = None
        
        
        if self.nms:
            result['dets'], result['det_types'], result['aux_info']['velos'] = \
                self.frame_nms(result['dets'], result['det_types'], result['aux_info']['velos'], self.nms_thres)
        result['dets'] = [BBox.bbox2array(d) for d in result['dets']]
        result['abs_frame_index'] = self.abs_frame_indices[self.cur_frame]

        self.cur_frame += 1
        return result
    
    def __len__(self):
        return self.max_frame
    
    def frame_nms(self, dets, det_types, velos, thres):
        frame_indexes, frame_types = nms(dets, det_types, thres)
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None
        if velos is not None:
            result_velos = [velos[i] for i in frame_indexes]
        return result_dets, frame_types, result_velos

    def reverse(self):
        self.dets_bboxes.reverse()
        self.dets_types.reverse()
        self.ego_info.reverse()
        self.ts_info.reverse()
        self.abs_frame_indices.reverse()
    
    def split(self, split_ratio):
        loader1 = copy.deepcopy(self)
        loader2 = copy.deepcopy(self)
        split = int(self.max_frame * split_ratio)

        loader1.dets_bboxes = loader1.dets_bboxes[:split]
        loader1.dets_types = loader1.dets_types[:split]
        loader1.ego_info = loader1.ego_info[:split]
        loader1.ts_info = loader1.ts_info[:split]
        loader1.abs_frame_indices = loader1.abs_frame_indices[:split]
        loader1.max_frame = split
        loader1.reverse()

        loader2.dets_bboxes = loader2.dets_bboxes[split:]
        loader2.dets_types = loader2.dets_types[split:]
        loader2.ego_info = loader2.ego_info[split:]
        loader2.ts_info = loader2.ts_info[split:]
        loader2.abs_frame_indices = loader2.abs_frame_indices[split:]
        loader2.max_frame = self.max_frame - split

        return loader1, loader2
