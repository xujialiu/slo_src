# distill_2layers_step1
nohup env CUDA_VISIBLE_DEVICES=3 python distill_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--clip_grad 1 \
--result_name "distill_2layers_step1" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--result_root_path "./model_testing_results/distill" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 2 \
--accum_iter 16 > ./model_testing_results/distill/distill_2layers_step1_`date +%Y%m%d_%H%M`.log 2>&1 &

# distill_4layers_step1
nohup env CUDA_VISIBLE_DEVICES=2 python distill_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--fusion_layer_num 4 \
--clip_grad 1 \
--result_name "distill_4layers_step1" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--result_root_path "./model_testing_results/distill" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 2 \
--accum_iter 16 > ./model_testing_results/distill/distill_4layers_step1_`date +%Y%m%d_%H%M`.log 2>&1 &


# distill_8layers_step1
nohup env CUDA_VISIBLE_DEVICES=2 python distill_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--fusion_layer_num 8 \
--clip_grad 1 \
--result_name "distill_8layers_step1" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--result_root_path "./model_testing_results/distill" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 2 \
--accum_iter 16 > ./model_testing_results/distill/distill_8layers_step1_`date +%Y%m%d_%H%M`.log 2>&1 &

# distill_16layers_step1
nohup env CUDA_VISIBLE_DEVICES=1 python distill_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--fusion_layer_num 16 \
--clip_grad 1 \
--result_name "distill_16layers_step1" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--result_root_path "./model_testing_results/distill" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 2 \
--accum_iter 16 > ./model_testing_results/distill/distill_16layers_step1_`date +%Y%m%d_%H%M`.log 2>&1 &

# resume distill_16layers_step1
nohup env CUDA_VISIBLE_DEVICES=3 python distill_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--resume "/data_A/xujialiu/projects/FMUE_SLO/model_testing_results/distill/distill_16layers_step1_20250417_1636/checkpoints/checkpoint_49.pth" \
--fusion_layer_num 16 \
--clip_grad 1 \
--result_name "resume_distill_16layers_step1" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--result_root_path "./model_testing_results/distill" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 1000 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--data_path "/home/xujia/datasets/DDR" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 2 \
--accum_iter 16 > ./model_testing_results/distill/resume_distill_16layers_step1_`date +%Y%m%d_%H%M`.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python distill_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type fp \
--resume "/data_A/xujialiu/projects/FMUE_SLO/model_testing_results/distill/distill_16layers_step1_20250417_1636/checkpoints/checkpoint_49.pth" \
--fusion_layer_num 16 \
--clip_grad 1 \
--result_name "resume_distill_16layers_step1" \
--num_patches "[(4,4),(3,3),(2,2),(1,1)]" \
--result_root_path "./model_testing_results/distill" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 1000 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR" \
--csv_path "/data_A/xujialiu/datasets/DDR/tabular_data/DDR_train_val_250408.csv" \
--input_size "(896,896)" \
--fm_input_size 224 \
--resize_to "(896,896)" \
--batch_size 2 \
--accum_iter 16


#----------
