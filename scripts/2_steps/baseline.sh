# baseline fusion_layer_num=2, 1step_training
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--clip_grad 1 \
--result_name "baseline_2layers" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./model_testing_results/2step" \
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
--accum_iter 16 > ./model_testing_results/2step/baseline_2layers_`date +%Y%m%d_%H%M`.log 2>&1 &


# baseline fusion_layer_num=2, 1step_training, fusion_num_heads=12, fusion_mlp_ratio=4,
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--clip_grad 1 \
--result_name "baseline_2layers_new" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./model_testing_results/2step" \
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
--batch_size 8 \
--accum_iter 4 > ./model_testing_results/2step/baseline_2layers_new_`date +%Y%m%d_%H%M`.log 2>&1 &


# baseline fusion_layer_num=4, 1step_training
nohup env CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--clip_grad 1 \
--fusion_layer_num 4 \
--result_name "baseline_4layers" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./model_testing_results/2step" \
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
--accum_iter 16 > ./model_testing_results/2step/baseline_4layers_`date +%Y%m%d_%H%M`.log 2>&1 &


# baseline fusion_layer_num=8, 1step_training
nohup env CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--clip_grad 1 \
--fusion_layer_num 8 \
--result_name "baseline_8layers" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./model_testing_results/2step" \
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
--batch_size 4 \
--accum_iter 8 > ./model_testing_results/2step/baseline_8layers_`date +%Y%m%d_%H%M`.log 2>&1 &


# baseline fusion_layer_num=8, 1step_training, fusion_num_heads=12, fusion_mlp_ratio=4
nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--clip_grad 1 \
--fusion_layer_num 8 \
--result_name "baseline_8layers_new" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./model_testing_results/2step" \
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
--batch_size 8 \
--accum_iter 4 > ./model_testing_results/2step/baseline_8layers_new_`date +%Y%m%d_%H%M`.log 2>&1 &


