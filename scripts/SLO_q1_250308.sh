# ori 224
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "SLO_visionfm_224_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./SLO_results/q1" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 0.9 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]" \
--batch_size 32 > ./SLO_logs/q1/SLO_visionfm_224_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# ori 448
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "SLO_visionfm_448_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./SLO_results/q1" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 0.9 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[448,448]" \
--fm_input_size 448 \
--resize_to "[448,448]" \
--batch_size 32 > ./SLO_logs/q1/SLO_visionfm_448_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# ori 672
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "SLO_visionfm_672_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./SLO_results/q1" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 0.9 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[672,672]" \
--fm_input_size 672 \
--resize_to "[672,672]" \
--batch_size 4 \
--accum_iter 8 > ./SLO_logs/q1/SLO_visionfm_672_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# ori 896
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "SLO_visionfm_896_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./SLO_results/q1" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 0.9 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[896,896]" \
--fm_input_size 896 \
--resize_to "[896,896]" \
--batch_size 32 > ./SLO_logs/q1/SLO_visionfm_896_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "SLO_visionfm_448_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./SLO_results/q1" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_large_patch16 \
--random_crop_perc 0.9 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[448,448]" \
--fm_input_size 448 \
--resize_to "[448,448]" \
--batch_size 32