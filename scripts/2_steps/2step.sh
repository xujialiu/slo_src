# fusion_layer_num=2, 1step_training
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--fusion_layer_num 2 \
--train_step 1 \
--clip_grad 1 \
--result_name "2step_1_2layers" \
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
--batch_size 32 > ./model_testing_results/2step/2step_1_2layers_`date +%Y%m%d_%H%M`.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--lora_position all \
--dataset_type fp \
--fusion_layer_num 2 \
--train_step 2 \
--clip_grad 1 \
--result_name "2step_2_2layers" \
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
--data_path "/data_A/xujialiu/datasets/DDR/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 8 \
--accum_iter 4 > ./model_testing_results/2step/2step_2_2layers_`date +%Y%m%d_%H%M`.log 2>&1 &

# fusion_layer_num=4, 1step_training
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--fusion_layer_num 4 \
--train_step 1 \
--clip_grad 1 \
--result_name "2step_1_4layers" \
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
--batch_size 32 > ./model_testing_results/2step/2step_1_4layers_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=2 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--lora_position all \
--dataset_type fp \
--fusion_layer_num 4 \
--train_step 2 \
--clip_grad 1 \
--result_name "2step_2_4layers" \
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
--data_path "/data_A/xujialiu/datasets/DDR/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 8 \
--accum_iter 4 > ./model_testing_results/2step/2step_2_4layers_`date +%Y%m%d_%H%M`.log 2>&1 &


# fusion_layer_num=8, 1step_training
nohup env CUDA_VISIBLE_DEVICES=3 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--fusion_layer_num 8 \
--train_step 1 \
--clip_grad 1 \
--result_name "2step_1_8layers" \
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
--data_path "/data_A/xujialiu/datasets/DDR/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 32 > ./model_testing_results/2step/2step_1_8layers_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=2 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--lora_position all \
--dataset_type fp \
--fusion_layer_num 8 \
--train_step 2 \
--clip_grad 1 \
--result_name "2step_2_8layers" \
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
--data_path "/data_A/xujialiu/datasets/DDR/preprocessed" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 8 \
--accum_iter 4 > ./model_testing_results/2step/2step_2_8layers_`date +%Y%m%d_%H%M`.log 2>&1 &