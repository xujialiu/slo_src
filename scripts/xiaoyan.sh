nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "850" \
--fulltune \
--result_root_path "./FP_results/xiaoyan" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_large_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/250318_xiaoyan/pretrain_all_public/checkpoint-850.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/xiaoyan/850_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "999" \
--fulltune \
--result_root_path "./FP_results/xiaoyan" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_large_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/250318_xiaoyan/pretrain_all_public/checkpoint-999.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/xiaoyan/999_`date +%Y%m%d_%H%M`.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "original" \
--fulltune \
--result_root_path "./FP_results/xiaoyan" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_large_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/RETFound_mae_natureCFP.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/xiaoyan/original_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "850" \
--fulltune \
--result_root_path "./FP_results/xiaoyan" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--warmup_epochs 3 \
--epochs 10 \
--model vit_large_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/250318_xiaoyan/pretrain_all_public/checkpoint-850.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/xiaoyan/850_`date +%Y%m%d_%H%M`.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--eval \
--result_name "850" \
--fulltune \
--result_root_path "./FP_results/xiaoyan_test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--warmup_epochs 3 \
--epochs 10 \
--model vit_large_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/FP_results/xiaoyan/850_20250319_1207/checkpoints/checkpoint_9.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 1

CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--eval \
--result_name "original" \
--fulltune \
--result_root_path "./FP_results/xiaoyan_test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_large_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/FP_results/xiaoyan/original_20250319_1207/checkpoints/checkpoint_6.pth" \
--data_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 1


# new 
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "stdr" \
--nb_classes 2 \
--clip_grad 1.0 \
--fulltune \
--result_root_path "./FP_results/xiaoyan" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/huxiaoyan/ProtoViT/data/STDR_macular_paired" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/xiaoyan/STDR_250328/stdr_data.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/xiaoyan/stdr_`date +%Y%m%d_%H%M`.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "stdr" \
--nb_classes 2 \
--fulltune \
--result_root_path "./FP_results/xiaoyan" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/huxiaoyan/ProtoViT/data/STDR_macular_paired" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/xiaoyan/STDR_250328/stdr_data.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4


# new 
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "stdr" \
--nb_classes 2 \
--clip_grad 1.0 \
--fulltune \
--result_root_path "./FP_results/xiaoyan" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 30 \
--warmup_epochs 5 \
--model vit_large_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/RETFound_mae_natureCFP.pth" \
--data_path "/data/huxiaoyan/ProtoViT/data/STDR_macular_paired" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/xiaoyan/STDR_250328/stdr_data.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/xiaoyan/stdr_`date +%Y%m%d_%H%M`.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--eval \
--result_name "stdr" \
--nb_classes 2 \
--clip_grad 1.0 \
--fulltune \
--result_root_path "./FP_results/xiaoyan_test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 30 \
--warmup_epochs 5 \
--model vit_large_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/FP_results/xiaoyan/stdr_20250328_1627/checkpoints/checkpoint_5.pth" \
--data_path "/data/huxiaoyan/ProtoViT/data/STDR_macular_paired" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/xiaoyan/STDR_250328/stdr_data.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/xiaoyan/stdr_`date +%Y%m%d_%H%M`.log 2>&1 &