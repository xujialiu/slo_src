# for test
CUDA_VISIBLE_DEVICES=2 python train_multiscale_energy.py \
--result_root_path "./SLO_results/stdr_bode_energy_250611" \
--result_name "stdr_bode_icdr_visionfm_energy_1344_4cls" \
--fusion_layer_num 8 \
--model vit_base_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(6,6), (5,5), (4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_datasets" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/250530_get_model_tabular_data/icdr4_250530.csv" \
--data_path_out "/data_A/xujialiu/datasets/UWF_other_diseases/ori" \
--csv_path_out "/data_A/xujialiu/datasets/UWF_other_diseases/250611_get_csv_for_energy/df_split.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32


nohup env CUDA_VISIBLE_DEVICES=2 python train_multiscale_energy.py \
--result_root_path "./SLO_results/stdr_bode_energy_250611" \
--result_name "stdr_bode_icdr_visionfm_energy_1344_4cls" \
--fusion_layer_num 8 \
--model vit_base_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(6,6), (5,5), (4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_datasets" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/250530_get_model_tabular_data/icdr4_250530.csv" \
--data_path_out "/data_A/xujialiu/datasets/UWF_other_diseases/ori" \
--csv_path_out "/data_A/xujialiu/datasets/UWF_other_diseases/250611_get_csv_for_energy/df_split.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_energy_250611/stdr_bode_icdr_visionfm_energy_1344_4cls_`date +%Y%m%d_%H%M`.log 2>&1 &



CUDA_VISIBLE_DEVICES=3 python train_multiscale_energy.py \
--eval \
--result_root_path "./SLO_results/stdr_bode_energy_250611" \
--result_name "test_eval" \
--fusion_layer_num 8 \
--model vit_base_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(6,6), (5,5), (4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_energy_250611/stdr_bode_icdr_visionfm_energy_1344_4cls_20250612_1021/checkpoints/checkpoint_26.pth" \
--data_path "/data_A/xujialiu/projects/FMUE_SLO/Denise/external_datasets/image_files" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/Denise/external_datasets/csv_files/Shuyi_DTS_study.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32