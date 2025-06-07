nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_multiscales_896_lora8_scale2_with_mean_1" \
--model vit_base_patch16_multiscales_with_mean \
--num_patches "[(4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/q5" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/home/xujia/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/q5/FP_visionfm_multiscales_896_lora8_scale2_with_mean_1_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_multiscales_896_lora8_scale2_multi_head_attention" \
--model vit_base_patch16_multiscales \
--num_patches "[(4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/q5" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/home/xujia/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/q5/FP_visionfm_multiscales_896_lora8_scale2_multi_head_attention_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=3 python train_patches.py \
--result_name "FP_visionfm_multiscales_896_lora8_scale2_mean_1_imagenet" \
--model vit_base_patch16_multiscales \
--num_patches "[(4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/q5" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/250312_get_vit_imagenet_ckpt/vit-base-imagenet.pth" \
--data_path "/home/xujia/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 8 \
--accum_iter 4 > ./FP_logs/q5/FP_visionfm_multiscales_896_lora8_scale2_mean_1_imagenet_`date +%Y%m%d_%H%M`.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python train_patches.py \
--result_name "FP_visionfm_multiscales_896_lora8_scale2_mean_1_imagenet" \
--model vit_base_patch16_multiscales \
--num_patches "[(4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/q5" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/250312_get_vit_imagenet_ckpt/vit-base-imagenet_0.pth" \
--data_path "/home/xujia/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 8