# 问题1: fulltune和lora4和lora8, 哪个效果好?

# fulltune
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_224_fulltune" \
--fulltune \
--result_root_path "./FP_results/q1" \
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
--resize_to "[1120,1120]" \
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# fulltune lrd 0.3
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_224_fulltune_lrd0.3" \
--fulltune \
--result_root_path "./FP_results/q1" \
--blr 5e-4 \
--layer_decay 0.3 \
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
--resize_to "[1120,1120]" \
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_fulltune_lrd0.3_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora4_scale2
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_224_lora4_scale2" \
--lora_rank 4 \
--lora_alpha 8 \
--result_root_path "./FP_results/q1" \
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
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_lora4_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora8_scale2
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_224_lora8_scale2" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/q1" \
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
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]" \
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_lora8_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora8_scale2 lora_dropout 0.2
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_224_lora8_scale2_dropout_0.2" \
--lora_rank 8 \
--lora_alpha 16 \
--lora_dropout 0.2 \
--result_root_path "./FP_results/q1" \
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
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]" \
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_lora8_scale2_dropout_0.2_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora8_scale2 lora_dropout 0.3
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "FP_visionfm_224_lora8_scale2_dropout_0.3" \
--lora_rank 8 \
--lora_alpha 16 \
--lora_dropout 0.3 \
--result_root_path "./FP_results/q1" \
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
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]" \
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_lora8_scale2_dropout_0.3_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora16 scale2
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_224_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/q1" \
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
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_lora16_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora16 scale2 lora_dropout 0.2
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_224_lora16_scale2_dropout_0.2" \
--lora_rank 16 \
--lora_alpha 32 \
--lora_dropout 0.2 \
--result_root_path "./FP_results/q1" \
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
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_lora16_scale2_dropout_0.2_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora16 scale2 lora_dropout 0.3
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_224_lora16_scale2_dropout_0.3" \
--lora_rank 16 \
--lora_alpha 32 \
--lora_dropout 0.3 \
--result_root_path "./FP_results/q1" \
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
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_lora16_scale2_dropout_0.3_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora32 scale2
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_224_lora32_scale2" \
--lora_rank 32 \
--lora_alpha 64 \
--result_root_path "./FP_results/q1" \
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
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_lora32_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora16 scale2 llrd
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_224_lora16_scale2_llrd" \
--lora_rank 16 \
--lora_alpha 32 \
--use_llrd \
--result_root_path "./FP_results/q1" \
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
--batch_size 32 > ./FP_logs/q1/FP_visionfm_224_lora16_scale2_llrd_`date +%Y%m%d_%H%M`.log 2>&1 &


#-----------test-----------#
CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "FP_visionfm_224_fulltune" \
--fulltune \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_test_20250305.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]" \
--batch_size 32


CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_224_lora16_scale2" \
--lora_rank 16 \
--lora_alpha 32 \
--result_root_path "./FP_results/test" \
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
--batch_size 32