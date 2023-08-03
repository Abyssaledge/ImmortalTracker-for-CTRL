bin_path=/mnt/weka/scratch/lve.fan/SST/data/important_bins/CTRL_FSD_TTA.bin
detection_name=CTRL_FSD_TTA # note the det name in training and validation is different, to keep consistent with vehicle
split=testing
python3 preparedata/waymo/detection.py --file_name $bin_path --name $detection_name --det_folder detection_data/processed --split $split