nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "SLO_uncertainty_1" \
--model vit_large_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/uncertainty" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "uncertainty" \
--csv_path "/data_A/xujialiu/datasets/daytona/dataset_tabular/gloabal_icdr_250322.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 > ./SLO_logs/uncertainty/SLO_uncertainty_1_`date +%Y%m%d_%H%M`.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "SLO_uncertainty_train_all" \
--train_all \
--model vit_large_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/uncertainty" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "uncertainty" \
--csv_path "/data_A/xujialiu/datasets/daytona/dataset_tabular/gloabal_icdr_train_all_250322.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 > ./SLO_logs/uncertainty/SLO_uncertainty_train_all_`date +%Y%m%d_%H%M`.log 2>&1 &

# 上面的代码有问题, EDL可能有问题
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "SLO_uncertainty_1" \
--model vit_large_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/uncertainty" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/daytona/dataset_tabular/gloabal_icdr_250322.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 > ./SLO_logs/uncertainty/SLO_uncertainty_1_`date +%Y%m%d_%H%M`.log 2>&1 &



# train_all
nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "SLO_uncertainty_train_all" \
--train_all \
--model vit_large_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/uncertainty" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/daytona/dataset_tabular/gloabal_icdr_train_all_250322.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 > ./SLO_logs/uncertainty/SLO_uncertainty_train_all_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "SLO_uncertainty_train_all_patches" \
--train_all \
--model vit_large_patch16_patches \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/uncertainty" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/daytona/dataset_tabular/gloabal_icdr_train_all_250322.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 > ./SLO_logs/uncertainty/SLO_uncertainty_train_all_patches_`date +%Y%m%d_%H%M`.log 2>&1 &

#-------------------------


CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "SLO_uncertainty_1" \
--model vit_large_patch16_patches \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/uncertainty" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/daytona/dataset_tabular/gloabal_icdr_train_all_250322.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 


CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "SLO_uncertainty_train_all" \
--train_all \
--model vit_large_patch16_multiscales \
--num_patches "[(2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "uncertainty" \
--csv_path "/data_A/xujialiu/datasets/daytona/dataset_tabular/gloabal_icdr_test_250322.csv" \
--input_size "[896,896]" \
--fm_input_size 448 \
--resize_to "[896,896]" \
--batch_size 2 \
--accum_iter 32



# 先用FP测试一下新的uncertainty loss
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_multiscales_672_lora8_scale2_uncertainty" \
--loss_type "uncertainty" \
--model vit_base_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/uncertainty" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR_IDRiD_messidor2" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[672,672]" \
--fm_input_size 224 \
--resize_to "[672,672]" \
--batch_size 16 \
--accum_iter 2 > ./FP_logs/uncertainty/FP_visionfm_multiscales_672_lora8_scale2_uncertainty_`date +%Y%m%d_%H%M`.log 2>&1 &

# 改用新的uncertainty
nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_multiscales_672_lora8_scale2_uncertainty_new" \
--loss_type "uncertainty_new" \
--model vit_base_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/uncertainty" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR_IDRiD_messidor2" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[672,672]" \
--fm_input_size 224 \
--resize_to "[672,672]" \
--batch_size 16 \
--accum_iter 2 > ./FP_logs/uncertainty/FP_visionfm_multiscales_672_lora8_scale2_uncertainty_new_`date +%Y%m%d_%H%M`.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_multiscales_672_lora8_scale2_uncertainty_new" \
--loss_type "uncertainty_new" \
--model vit_base_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/uncertainty" \
--blr 2e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR_IDRiD_messidor2" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[672,672]" \
--fm_input_size 224 \
--resize_to "[672,672]" \
--batch_size 16 \
--accum_iter 2 > ./FP_logs/uncertainty/FP_visionfm_multiscales_672_lora8_scale2_uncertainty_new_`date +%Y%m%d_%H%M`.log 2>&1 &

# 降低学习率
nohup env CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_multiscales_672_lora8_scale2_uncertainty" \
--loss_type "uncertainty" \
--model vit_base_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/uncertainty" \
--blr 2e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR_IDRiD_messidor2" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250305.csv" \
--input_size "[672,672]" \
--fm_input_size 224 \
--resize_to "[672,672]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/uncertainty/FP_visionfm_multiscales_672_lora8_scale2_uncertainty_`date +%Y%m%d_%H%M`.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_multiscales_672_lora8_scale2_uncertainty" \
--loss_type "uncertainty" \
--model vit_base_patch16_multiscales \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR_IDRiD_messidor2" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_test_20250305.csv" \
--input_size "[672,672]" \
--fm_input_size 224 \
--resize_to "[672,672]" \
--batch_size 4 \
--accum_iter 2

