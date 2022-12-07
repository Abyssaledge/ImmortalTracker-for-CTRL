bin_path=detection_data/waymo_bins/fsd_jrnl_12e_7f_6base_pd_gamma2_unet_rerun_val.bin
detection_name=fsdpp
python preparedata/waymo/detection.py --file_name $bin_path --name $detection_name --det_folder detection_data/processed