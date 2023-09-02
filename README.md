# Immortal_tracker for CTRL
## This is a modified unofficial version of [ImmortalTracker](https://github.com/ImmortalTracker/ImmortalTracker), tailored to easier use for CTRL. We make the follow modifications:

- Calculate 3D IoU on GPU for much faster tracking, requiring MMDet3D (<v1.0).
- Support saving the extended boxes (by motion) for the forward tracking part in CTRL.
- Improve readability.
- Automatically evaluation on WOD, requiring Waymo evaluation tools.

These modifications is showcased in the config file: `configs/immortal_for_ctrl_keep10.yaml`. Users could compare it with official config to understand the differences.

Please follow the official instruction of ImmortalTracker to prepare the data, and then user could run `run_mot.sh` to get the tracking results. Note you may need to rename some variables in `run_mot.sh` to fit your needs.

Feel free to open an issue or contact me (fanlue2019@ia.ac.cn) if you have any problem.

# Original README as follows
--- 

## Prerequisite
Our code is tested for Python 3.6.\
To install required liabraries:
```
pip install -r requirements.txt
```

## Waymo Open Dataset
### Prepare dataset & off-the-shelf detections
#### Download WOD perception dataset:
```
#Waymo Dataset         
└── waymo
       ├── training (not required)  
       ├── validation   
       ├── testing 
```
To extract timestamp infos/ego infos from .tfrecord files, run the following:
```
bash preparedata/waymo/waymo_preparedata.sh  /<path to WOD>/waymo
```
Run the following to convert detection results into to .npz files. The detection results should be in official WOD submission format(.bin)  
We recommand you to use CenterPoint(two-frame model for tracking) detection results for reproducing our results. Please follow https://github.com/tianweiy/CenterPoint or email its author for CenterPoint detection results.
```
bash preparedata/waymo/waymo_convert_detection.sh <path to detection results>/detection_result.bin cp

#you can also use other detections:
#bash preparedata/waymo/waymo_convert_detection.sh <path to detection results> <detection name>
```


### Inference
Use the following command to start inferencing on WOD. The validation set is used by default.
```
python main_waymo.py --name immortal --det_name cp --config_path configs/waymo_configs/immortal.yaml --process 8
```

### Evaluation with WOD official devkit:
Follow https://github.com/waymo-research/waymo-open-dataset to build the evaluation tools and run the following command for evaluation:
```
#Convert the tracking results into .bin file
python evaluation/waymo/pred_bin.py --name immortal
#For evaluation
<path to WOD devkit>/bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main mot_results/waymo/validation/immortal/bin/pred.bin <path to WOD tracking ground truth file>/validation_gt.bin
```



## nuScenes Dataset
### Prepare dataset & off-the-shelf detections
#### Download nuScenes perception dataset
```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       
       ├── sweeps       
       ├── maps         
       ├── v1.0-trainval 
       ├── v1.0-test
```
To extract timestamp infos/ego infos, run the following:

```
bash preparedata/nuscenes/nu_preparedata.sh <path to nuScenes>/nuscenes
```

Run the following to convert detection results into to .npz files. The detection results should be in official nuScenes submission format(.json)  
We recommand you to use centerpoint(two-frame model for tracking) detection results for reproducing our results.
```
bash preparedata/nuscenes/nu_convert_detection.sh  <path to detection results>/detection_result.json cp

#you can also use other detections:
#bash preparedata/nuscenes/nu_convert_detection.sh <path to detection results> <detection name>
```

### Inference
Use the following command to start inferencing on nuScenes. The validation set is used by default.
```
python main_nuscenes.py --name immortal --det_name cp --config_path configs/nu_configs/immortal.yaml --process 8
```

### Evaluation with nuScenes official devkit:
Follow https://github.com/nutonomy/nuscenes-devkit to build the official evaluation tools for nuScenes. Run the following command for evaluation:
```
#To convert tracking results into .json format
bash evaluation/nuscenes/pipeline.sh immortal
#To evaluate
python <path to nuscenes-devkit>/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/evaluate.py \
"./mot_results/nuscenes/validation_2hz/immortal/results/results.json" \
--output_dir "./mot_results/nuscenes/validation_2hz/immortal/results" \
--eval_set "val" \
--dataroot <path to nuScenes>/nuscenes
```

