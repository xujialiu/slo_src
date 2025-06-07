# 问题2: patches 896和直接896, 哪个效果好?

# 统一使用lora16_scale2

# ori 896
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
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
--batch_size 2 \
--accum_iter 16  > ./FP_logs/q1/FP_visionfm_896_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &


# 使用patches 896
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16_patches \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q2" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 2 \
--accum_iter 16 > ./FP_logs/q2/FP_visionfm_patches_896_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &



# 使用patches 896, 加上patch_pos_embed
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2_with_ppe" \
--model vit_base_patch16_multiscales \
--num_patches "[(4,4)]" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q2" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 2 \
--accum_iter 16 > ./FP_logs/q2/FP_visionfm_patches_896_lora16_scale2_with_ppe_`date +%Y%m%d_%H%M`.log 2>&1 &


# 使用multiscales 896 [(4,4), (2,2), (1,1)]
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "FP_visionfm_multiscales_896_lora16_scale2" \
--model vit_base_patch16_multiscales \
--num_patches "[(4,4), (3,3),(2,2), (1,1)]" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q2" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 2 \
--accum_iter 16 > ./FP_logs/q2/FP_visionfm_multiscales_896_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# 使用multiscales 896 [(4,4), (2,2), (1,1)] without ppe
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_multiscales_896_lora16_scale2_without_ppe" \
--model vit_base_patch16_multiscales \
--num_patches "[(4,4), (2,2), (1,1)]" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q2" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 2 \
--accum_iter 16 > ./FP_logs/q2/FP_visionfm_multiscales_896_lora16_scale2_without_ppe_`date +%Y%m%d_%H%M`.log 2>&1 &

# ----------test---------- #




