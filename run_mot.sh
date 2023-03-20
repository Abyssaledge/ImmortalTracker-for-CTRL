# config=configs/waymo_configs/tl_basic.yaml
# name=tl_basic
config=configs/waymo_configs/my_immortal.yaml
name=my_immortal
# det_name=fsd_trainonly_paint_notta
det_name=CTRL_FSD_TTA
# det_name=fsd_pastfuture_12tta
# det_name=fsd_pastfuture
# obj_type=vehicle
# obj_type=pedestrian
obj_type=cyclist
process=32
# split=training
# split=validation
split=testing
python3 main_waymo.py --name $name --det_name $det_name --config_path $config --process $process --det_data_folder ./detection_data/processed --obj_type $obj_type --split $split

# obj_types=vehicle
# obj_types=pedestrian
obj_types=cyclist
python3 evaluation/waymo/pred_bin.py --name $name --det_name $det_name --config_path $config --split $split --obj_types $obj_types #--no-eval #--output_file_name pred_maxtime5
