# ori 448
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_448_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q4" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[448,448]" \
--fm_input_size 448 \
--resize_to "[448,448]" \
--batch_size 32 > ./FP_logs/q4/FP_visionfm_448_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# ori 672
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "FP_visionfm_672_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q4" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[672,672]" \
--fm_input_size 672 \
--resize_to "[672,672]" \
--batch_size 4 \
--accum_iter 8 > ./FP_logs/q4/FP_visionfm_672_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# ori 896 在q2中
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_896_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q4" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[896,896]" \
--fm_input_size 896 \
--resize_to "[896,896]" \
--batch_size 32 > ./FP_logs/q4/FP_visionfm_896_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &


# ori 224 在q1中
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_224_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q4" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]" \
--batch_size 32 > ./FP_logs/q4/FP_visionfm_224_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &