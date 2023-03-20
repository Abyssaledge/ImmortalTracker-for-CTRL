# bin_path=detection_data/waymo_bins/fsd_jrnl_12e_7f_6base_pd_gamma2_unet_rerun_val.bin
# bin_path=detection_data/waymo_bins/fsd_6f_6e_valset.bin
# bin_path=detection_data/waymo_bins/fsd_9f_iter2.bin
# bin_path=detection_data/waymo_bins/fsd_pastfuture_veh_refined1_trainset.bin
# bin_path=detection_data/waymo_bins/fsd_pastfuture_testset.bin
# bin_path=/mnt/weka/scratch/lve.fan/SST_release/work_dirs/fsd_9f_trainval/results_testset_12tta.bin
# bin_path=/mnt/weka/scratch/lve.fan/SST_release/work_dirs/fsd_waymoD1_1x_futuresweeps_framedrop/results_val_12tta.bin
# bin_path=/mnt/weka/scratch/lve.fan/SST/work_dirs_trk/fsd_9f_painting/results_test_no_tta.bin
bin_path=/mnt/weka/scratch/lve.fan/SST/data/important_bins/CTRL_FSD_TTA.bin
detection_name=CTRL_FSD_TTA # note the det name in training and validation is different, to keep consistent with vehicle
# split=training
# split=validation
split=testing
python3 preparedata/waymo/detection.py --file_name $bin_path --name $detection_name --det_folder detection_data/processed --split $split