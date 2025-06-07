nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "SLO3" \
--fulltune \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250225.csv" \
--input_size "[1344,1792]" \
--device "cuda:0" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32 > ./logs/slo3_`date +%Y%m%d_%H%M`.log 2>&1 &




CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--eval \
--result_name "eval_SLO2_20250301_1048_epoch_61_bode" \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/results/SLO2_20250301_1048/checkpoints/checkpoint-best_61.pth" \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/validation_tabular/bode_250304.csv" \
--input_size "[1344,1792]" \
--device "cuda:0" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32
CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--eval \
--result_name "eval_SLO2_20250301_1048_epoch_61_UWF4DR" \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/results/SLO2_20250301_1048/checkpoints/checkpoint-best_61.pth" \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/UWF4DR2024/UWF4DR2024_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/validation_tabular/UWF4DR2024_task2_250304.csv" \
--input_size "[1344,1792]" \
--device "cuda:0" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32


CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--eval \
--result_name "eval_SLO2_20250228_1033_epoch_52_bode" \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/results/SLO2_20250228_1033/checkpoints/checkpoint-best_52.pth" \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/validation_tabular/bode_250304.csv" \
--input_size "[1792,1792]" \
--device "cuda:0" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32
CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--eval \
--result_name "eval_SLO2_20250228_1033_epoch_52_UWF4DR" \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/results/SLO2_20250228_1033/checkpoints/checkpoint-best_52.pth" \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/UWF4DR2024/UWF4DR2024_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/validation_tabular/UWF4DR2024_task2_250304.csv" \
--input_size "[1792,1792]" \
--device "cuda:0" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32


CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--eval \
--result_name "eval_SLO3_20250304_1745_epoch_19" \
--fulltune \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/results/SLO3_20250304_1745/checkpoints/checkpoint-best_11.pth" \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/validation_tabular/bode_250304.csv" \
--input_size "[1344,1792]" \
--device "cuda:0" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32
