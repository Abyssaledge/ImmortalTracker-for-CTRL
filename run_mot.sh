name=immortal_test_refactor
det_name=fsdpp
config=configs/waymo_configs/immortal.yaml
obj_type=vehicle
process=32
python3 main_waymo.py --name $name --det_name $det_name --config_path $config --process $process --det_data_folder ./detection_data/processed --obj_type $obj_type
python3 evaluation/waymo/pred_bin.py --name $name --det_name $det_name
./compute_tracking_metrics_main_detail ./mot_results/waymo/validation/${name}_$det_name/bin/pred.bin gt.bin
