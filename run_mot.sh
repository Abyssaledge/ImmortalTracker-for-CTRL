config=configs/waymo_configs/immortal_for_ctrl_keep10.yaml
name=immortal
det_name=CTRL_FSD_TTA
obj_type=vehicle
# obj_type=pedestrian
# obj_type=cyclist
process=32
# split=training
# split=validation
split=testing
python3 main_waymo.py --name $name --det_name $det_name --config_path $config --process $process --det_data_folder ./detection_data/processed --obj_type $obj_type --split $split

obj_types=vehicle
# obj_types=pedestrian
# obj_types=cyclist
python3 evaluation/waymo/pred_bin.py --name $name --det_name $det_name --config_path $config --split $split --obj_types $obj_types --no-eval
