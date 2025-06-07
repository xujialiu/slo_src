CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16_multiscales \
--num_patches "[(2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[448,448]" \
--fm_input_size 224 \
--resize_to "[448,448]" \
--batch_size 1

CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16_multiscales \
--num_patches "[(3,3),(2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[672,672]" \
--fm_input_size 224 \
--resize_to "[672,672]" \
--batch_size 1

CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16_multiscales \
--num_patches "[(4,4), (3,3),(2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 1

CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16_multiscales \
--num_patches "[(5,5), (4,4), (3,3),(2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[1120,1120]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 1


# original
CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16 \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]" \
--batch_size 1


CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16 \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[448,448]" \
--fm_input_size 224 \
--resize_to "[448,448]" \
--batch_size 1


CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16 \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[672,672]" \
--fm_input_size 224 \
--resize_to "[672,672]" \
--batch_size 1


CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16 \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 1


CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16 \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[1120,1120]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 1

# patches
CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16_patches \
--num_patches "[(2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[448,448]" \
--fm_input_size 224 \
--resize_to "[448,448]" \
--batch_size 1

CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16_patches \
--num_patches "[(3,3),(2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[672,672]" \
--fm_input_size 224 \
--resize_to "[672,672]" \
--batch_size 1

CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16_patches \
--num_patches "[(4,4), (3,3),(2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[896,896]" \
--fm_input_size 224 \
--resize_to "[896,896]" \
--batch_size 1

CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "FP_visionfm_patches_896_lora16_scale2" \
--model vit_base_patch16_patches \
--num_patches "[(5,5), (4,4), (3,3),(2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./FP_results/test" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data/xujialiu/datasets/DDR_IDRiD_messidor2" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/DDR_IDRiD_messidor2/DDR_IDRiD_messidor2_250310.csv" \
--input_size "[1120,1120]" \
--fm_input_size 224 \
--resize_to "[1120,1120]" \
--batch_size 1
