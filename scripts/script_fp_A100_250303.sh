# patches lora baseline
# FP_patches_visionfm_1120
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--model vit_base_patch16_patches \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_patches_visionfm_1120" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[1120,1120]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 16 \
--accum_iter 2 > ./logs/FP_patches_visionfm_1120_`date +%Y%m%d_%H%M`.log 2>&1 &

# 尝试使用patches模型的fulltune, 结论:
# FP_patches_visionfm_1120_fulltune_20250304_1638
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--model vit_base_patch16_patches \
--random_crop_perc 1 \
--fulltune \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_patches_visionfm_1120_fulltune" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[1120,1120]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4 > ./logs/FP_patches_visionfm_1120_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# 非patches lora baseline
# FP_visionfm_224_20250303_1413
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_visionfm_224" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 32 > ./logs/FP_visionfm_224_`date +%Y%m%d_%H%M`.log 2>&1 &

# 尝试将blr降为原来的1/2, 结论: 效果不好
# FP_visionfm_224_20250303_1706
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--model vit_base_patch16 \
--random_crop_perc 1 \
--blr 1e-3 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_visionfm_224" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 32 > ./logs/FP_visionfm_224_`date +%Y%m%d_%H%M`.log 2>&1 &

# 尝试使用非patches模型的fulltune, 结论:
# FP_visionfm_224_fulltune_20250304_1622
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--model vit_base_patch16 \
--random_crop_perc 1 \
--fulltune \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_visionfm_224_fulltune" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 32 > ./logs/FP_visionfm_224_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# 尝试将fulltune的blr降为原来的1/2, 结论:
# FP_visionfm_224_fulltune_20250304_1642
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--model vit_base_patch16 \
--random_crop_perc 1 \
--fulltune \
--blr 1e-3 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_visionfm_224_fulltune" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 32 > ./logs/FP_visionfm_224_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# 尝试将fulltune的blr降为原来的1/4, 结论:
# FP_visionfm_224_fulltune_20250304_1643
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--model vit_base_patch16 \
--random_crop_perc 1 \
--fulltune \
--blr 5e-4 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_visionfm_224_fulltune" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 32 > ./logs/FP_visionfm_224_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# 尝试将lora_rank设置为8, 结论:
# FP_visionfm_224_lora8_20250304_1708
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--model vit_base_patch16 \
--random_crop_perc 1 \
--lora_rank 8 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_visionfm_224_lora8" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 32 > ./logs/FP_visionfm_224_lora8_`date +%Y%m%d_%H%M`.log 2>&1 &


# FP_retfound_224_20250303_1449
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--model vit_large_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/RETFound_mae_natureCFP.pth" \
--result_name "FP_retfound_224" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 32 > ./logs/FP_retfound_224_`date +%Y%m%d_%H%M`.log 2>&1 &

# 尝试非patches模型使用fulltune, 结论: 效果比lora好
# FP_visionfm_224_fulltune_20250303_1651
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--model vit_base_patch16 \
--random_crop_perc 1 \
--fulltune \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_visionfm_224_fulltune" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 32 > ./logs/FP_visionfm_224_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# 尝试patches模型使用fulltune
# FP_patches_visionfm_1120_fulltune_20250304_0957
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--model vit_base_patch16_patches \
--random_crop_perc 1 \
--fulltune \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "FP_patches_visionfm_1120_fulltune" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250303.csv" \
--input_size "[1120,1120]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4 > ./logs/FP_patches_visionfm_1120_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# test
CUDA_VISIBLE_DEVICES=2 python train_patches.py \
--model vit_base_patch16_patches \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--result_name "test" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_test_20250304.csv" \
--input_size "[1120,1120]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 16 \
--accum_iter 2
