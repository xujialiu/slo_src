# SLO2_20250225_1057 baseline
nohup python train_patches.py \
--result_name "SLO2" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250225.csv" \
--input_size "[1344,1792]" \
--device "cuda:0" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32 > ./logs/slo2_`date +%Y%m%d_%H%M`.log 2>&1 &

# SLO2_20250225_1100 SLO2_20250301_1048 测试fusion_dropout=0
nohup python train_patches.py \
--result_name "SLO2" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250225.csv" \
--input_size "[1344,1792]" \
--device "cuda:1" \
--batch_size 1 \
--fusion_dropout 0 \
--accum_iter 32 > ./logs/slo2_`date +%Y%m%d_%H%M`.log 2>&1 &

# SLO2_20250228_1033 测试将input_size改为[1792,1792]
nohup python train_patches.py \
--result_name "SLO2" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250225.csv" \
--input_size "[1792,1792]" \
--device "cuda:0" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32 > ./logs/slo2_`date +%Y%m%d_%H%M`.log 2>&1 &

# SLO2_20250301_1048, 测试不使用cross entropy weight
nohup python train_patches.py \
--result_name "SLO2" \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250225.csv" \
--input_size "[1344,1792]" \
--device "cuda:3" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32 > ./logs/slo2_`date +%Y%m%d_%H%M`.log 2>&1 &

# 测试使用fulltune
# SLO2_20250303_1402, 代码有问题
# SLO2_20250304_1533
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "SLO2" \
--fulltune \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250225.csv" \
--input_size "[1344,1792]" \
--device "cuda:0" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32 > ./logs/slo2_`date +%Y%m%d_%H%M`.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--eval \
--result_name "eval" \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/results/SLO2_20250301_1048/checkpoints/checkpoint-best_61.pth" \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/validation_tabular/bode_250304.csv" \
--input_size "[1344,1792]" \
--device "cuda:0" \
--batch_size 10 \
--fusion_dropout 0.1 \
--accum_iter 32



CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--eval \
--result_name "eval_1792*1792" \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/results/SLO2_20250228_1033/checkpoints/checkpoint-best_52.pth" \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/validation_tabular/bode_250304.csv" \
--input_size "[1792,1792]" \
--device "cuda:0" \
--batch_size 10 \
--fusion_dropout 0.1 \
--accum_iter 32


CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--eval \
--result_name "eval" \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/results/SLO2_20250303_1402/checkpoints/checkpoint-best_13.pth" \
--use_cross_entropy_weight False \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/validation_tabular/bode_250304.csv" \
--input_size "[1344,1792]" \
--device "cuda:0" \
--batch_size 10 \
--fusion_dropout 0.1 \
--accum_iter 32

# test
python train_patches.py \
--result_name "test2" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--random_crop_perc 1 \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/test_20250213.csv" \
--input_size "[896,896]" \
--resize_to "[896,896]" \
--fm_input_size 448 \
--device "cuda:2" \
--batch_size 1 \
--fusion_dropout 0 \
--accum_iter 32

