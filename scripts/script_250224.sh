# 使用train.py
nohup python train.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size 448 \
--device "cuda:0" \
--batch_size 8 > ./logs/slo_`date +%Y%m%d_%H%M%S`.log 2>&1 &

# 使用train_new.py
nohup python train_new.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size 448 \
--device "cuda:0" \
--batch_size 8 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &

# 使用train_new.py, 尝试增加分辨率是否有助于提高正确率
nohup python train_new.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size 896 \
--device "cuda:0" \
--batch_size 1 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &


tensorboard --logdir="results/SLO_20250215_1034/log" --port=10000

tensorboard --logdir="results/SLO_20250213_150459/log" --port=12126
tensorboard --logdir="results/SLO_20250213_130244/log" --port=12125
tensorboard --logdir="results/SLO_20250213_112220/log" --port=12124
tensorboard --logdir="results/SLO_20250213_100140/log" --port=12123
tensorboard --logdir="results/SLO_20250212_114053/log" --port=12122
tensorboard --logdir="/data_A/xujialiu/projects/FMUE_SLO/results/SLO2_20250225_1057/log" --port=10000
tensorboard --logdir="/data_A/xujialiu/projects/FMUE_SLO/results/SLO2_20250225_1100/log" --port=20000

# 用于测试的代码
python train_new.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/test_20250213.csv" \
--input_size 896 \
--device "cuda:0" \
--batch_size 1



nohup python train_patches.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size "1344,1792" \
--device "cuda:0" \
--batch_size 1 \
--accum_iter 16 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &



tensorboard --logdir="/data_A/xujialiu/projects/FMUE_SLO/results/SLO_20250223_1528/log" --port=20000


nohup python train_patches.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size "1344,1792" \
--device "cuda:0" \
--batch_size 2 \
--accum_iter 16 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &


# 2025.02.21
nohup python train_patches.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size "1344,1792" \
--device "cuda:0" \
--batch_size 1 \
--lora_dropout 0 \
--accum_iter 32 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &


# 尝试不使用数据增广
nohup python train_patches.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--loss_type "cross_entropy" \
--use_augmentation "False" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size "1344,1792" \
--device "cuda:1" \
--batch_size 1 \
--lora_dropout 0.1 \
--accum_iter 32 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup python train_patches.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250225.csv" \
--input_size "[1344,1792]" \
--device "cuda:1" \
--batch_size 1 \
--fusion_dropout 0.1 \
--accum_iter 32 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &


# 尝试去掉fusion的dropout
nohup python train_patches.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--loss_type "cross_entropy" \
--use_augmentation "False" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/filtered_20250211.csv" \
--input_size "1344,1792" \
--device "cuda:1" \
--batch_size 1 \
--fusion_dropout 0 \
--accum_iter 32 > ./logs/slo_`date +%Y%m%d_%H%M`.log 2>&1 &


# testing
python train_patches.py \
--data_path "/data_A/xujialiu/datasets/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/test_20250213.csv" \
--input_size "896,896" \
--device "cuda:1" \
--batch_size 1 \
--accum_iter 32


# 测试patches模型
python train_patches.py \
--result_name "test" \
--data_path "/data_A/xujialiu/datasets/daytona" \
--loss_type "uncertainty" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/test_20250213.csv" \
--input_size "[896,896]" \
--device "cuda:1" \
--batch_size 1 \
--accum_iter 32

# 测试原始模型
python train_patches.py \
--model vit_large_patch16 \
--result_name "test" \
--data_path "/data_A/xujialiu/datasets/daytona" \
--loss_type "uncertainty" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/test_20250213.csv" \
--input_size "[448,448]" \
--device "cuda:3" \
--batch_size 1 \
--accum_iter 32