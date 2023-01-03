config=configs/waymo_configs/immortal_hit0_score0_mul09.yaml
name=immortal_hit0_score0_mul09
det_name=fsdpp
obj_type=vehicle
process=32
python3 main_waymo.py --name $name --det_name $det_name --config_path $config --process $process --det_data_folder ./detection_data/processed --obj_type $obj_type
python3 evaluation/waymo/pred_bin.py --name $name --det_name $det_name --config_path $config --output_file_name pred_maxtime5
