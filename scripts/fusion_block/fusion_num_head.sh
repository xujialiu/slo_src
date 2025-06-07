# num_head_16
nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--fusion_num_head 16 \
--clip_grad 1 \
--result_name "num_head_16" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./model_testing_results/num_head" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 2 \
--accum_iter 16 > ./model_testing_results/num_head/num_head_16_`date +%Y%m%d_%H%M`.log 2>&1 &

# num_head_12
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--fusion_num_head 12 \
--clip_grad 1 \
--result_name "num_head_12" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./model_testing_results/num_head" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 2 \
--accum_iter 16 > ./model_testing_results/num_head/num_head_12_`date +%Y%m%d_%H%M`.log 2>&1 &

# num_head_8
nohup env CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--fusion_num_head 8 \
--clip_grad 1 \
--result_name "num_head_8" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./model_testing_results/num_head" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 2 \
--accum_iter 16 > ./model_testing_results/num_head/num_head_8_`date +%Y%m%d_%H%M`.log 2>&1 &