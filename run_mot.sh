name=immortal
det_name=fsdpp
config=immortal.yaml
obj_type=pedestrian
python main_waymo.py --name $name --det_name $det_name --config_path configs/waymo_configs/$config --process 99 --det_data_folder ./detection_data/processed --obj_type $obj_type
python evaluation/waymo/pred_bin.py --name $name --det_name $det_name
./compute_tracking_metrics_main_official ./mot_results/waymo/validation/${name}_$det_name/bin/pred.bin ../transdet3d/gt.bin