nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--result_name "stdr_bode_icdr_deepuwf_1344" \
--model vit_large_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250423" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_icdr_250423.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250423/stdr_bode_icdr_deepuwf_1344_`date +%Y%m%d_%H%M`.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--result_name "stdr_bode_icdr_visionfm_2240" \
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
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_icdr_250423.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250423/stdr_bode_icdr_visionfm_2240_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--result_name "stdr_bode_icdr01_visionfm_2240" \
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
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_icdr_01_250501.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250423/stdr_bode_icdr01_visionfm_2240_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--result_name "stdr_bode_icdr01_visionfm_2240_combine_01_add_nv_prh" \
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
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_icdr_combine_01_add_nv_prh_250503.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250423/stdr_bode_icdr01_visionfm_2240_combine_01_add_nv_prh_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--result_name "stdr_bode_dme_visionfm_2240" \
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
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path /data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_dme_250505.csv \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250423/stdr_bode_dme_visionfm_2240_`date +%Y%m%d_%H%M`.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--result_name "stdr_bode_visionfm_2240_8layer_add_nv_prh" \
--model vit_base_patch16_multiscales \
--fusion_layer_num 8 \
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
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_icdr_add_nv_prh_250509.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250423/stdr_bode_visionfm_2240_8layer_add_nv_prh_`date +%Y%m%d_%H%M`.log 2>&1 &




CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--eval \
--result_name "eval_stdr_bode_icdr_visionfm_2240" \
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
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_icdr_add_nv_prh_combine_01_val_250509.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32

CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--eval \
--result_name "eval_stdr_bode_icdr_visionfm_2240" \
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
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250423/stdr_bode_icdr_visionfm_2240_20250428_1447/checkpoints/checkpoint_33.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/UWF_DR/UWF_DR" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_external/UWF_DR/UWF_DR_tabular/UWF_DR_filtered_pseudo_label_250501.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32

CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--eval \
--result_name "eval_stdr_bode_icdr_visionfm_2240" \
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
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250423/stdr_bode_icdr_visionfm_2240_20250428_1447/checkpoints/checkpoint_19.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/Tsukazaki" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_external/000000_prepare_external_validation/external_validation.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32



CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--eval \
--result_name "eval_stdr_bode_icdr_visionfm_2240" \
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
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250423/stdr_bode_icdr01_visionfm_2240_20250501_1626/checkpoints/checkpoint_20.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/Tsukazaki" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_external/000000_prepare_external_validation_01/external_validation.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32


CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--eval \
--result_name "eval_Tsukazaki_old" \
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
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250423/stdr_bode_icdr_visionfm_2240_20250428_1447/checkpoints/checkpoint_19.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/Tsukazaki" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_external/000000_prepare_external_validation/Tsukazaki_5.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32


CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--eval \
--result_name "eval_!" \
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
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/test_other_diseases/test_dme.csv" \
--input_size "[2240,2240]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32