# GDPH_UWF
CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--eval \
--result_name "eval_GDPH_UWF" \
--model vit_base_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(10,10),(9,9),(8,8),(7,7),(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250423" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250423/stdr_bode_icdr01_visionfm_2240_combine_01_add_nv_prh_20250504_1155/checkpoints/checkpoint_29.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/GDPH_UWF" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_external/000000_prepare_external_validation/GDPH_UWF.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32

# GDPH_UWF_01
CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--eval \
--result_name "eval_GDPH_UWF_01" \
--model vit_base_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(10,10),(9,9),(8,8),(7,7),(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250423" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 2 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250423/stdr_bode_icdr01_visionfm_2240_20250501_1626/checkpoints/checkpoint_30.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/GDPH_UWF" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_external/000000_prepare_external_validation/GDPH_UWF_01.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32

# GDPH_UWF_DME
CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--eval \
--result_name "eval_GDPH_UWF_DME" \
--model vit_base_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(10,10),(9,9),(8,8),(7,7),(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250423" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 2 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250423/stdr_bode_dme_visionfm_2240_20250507_1538/checkpoints/checkpoint_12.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/GDPH_UWF" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_external/000000_prepare_external_validation/GDPH_UWF_DME.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32