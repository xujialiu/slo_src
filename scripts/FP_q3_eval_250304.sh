# 问题3: patches的增加和performance的关系

# 使用multiscales 448 [(2,2), (1,1)]
CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--eval \
--result_name "FP_visionfm_multiscales_448_lora16_scale2" \
--model vit_base_patch16_multiscales \
--num_patches "[(2,2), (1,1)]" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q3_eval" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[448,448]" \
--fm_input_size 224 \
--resize_to "[448,448]" \
--batch_size 2 \
--accum_iter 16 

# 使用multiscales 672 [(3,3), (2,2), (1,1)]
CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--eval \
--result_name "FP_visionfm_multiscales_672_lora16_scale2" \
--model vit_base_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q3_eval" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[672,672]" \
--fm_input_size 224 \
--resize_to "[672,672]" \
--batch_size 2 \
--accum_iter 16 

# 使用multiscales 896 [(4,4), (3,3), (2,2), (1,1)]
CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--eval \
--result_name "FP_visionfm_multiscales_896_lora16_scale2" \
--model vit_base_patch16_multiscales \
--num_patches "[(4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q3_eval" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/FP_results/q3/FP_visionfm_multiscales_896_lora16_scale2_20250309_1611/checkpoints/checkpoint_13.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 8

# 使用multiscales 1120 [(5,5), (4,4), (3,3), (2,2), (1,1)]
CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--eval \
--result_name "FP_visionfm_multiscales_1120_lora16_scale2" \
--model vit_base_patch16_multiscales \
--num_patches "[(5,5), (4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q3_eval" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[1120,1120]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 2 \
--accum_iter 16