# deepuwf 4 classes
CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--eval \
--result_name "eval_deepuwf_GDPH_UWF" \
--fusion_layer_num 8 \
--model vit_large_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250510" \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250510/stdr_bode_icdr_deepuwf_1344_4cls_20250510_1652/checkpoints/checkpoint_25.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/GDPH_UWF" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_external/000000_prepare_external_validation/GDPH_UWF.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--batch_size 1

# visionfm 4 classes
CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--eval \
--result_name "eval_visionfm_GDPH_UWF" \
--fusion_layer_num 8 \
--model vit_base_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(6,6), (5,5), (4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250510" \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250510/stdr_bode_icdr_visionfm_1344_4cls_20250513_1814/checkpoints/checkpoint_38.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/GDPH_UWF" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_external/000000_prepare_external_validation/GDPH_UWF.csv" \--input_size "[1344,1344]" \
--fm_input_size 224 \
--batch_size 1


# deepuwf 5 classes
nohup env CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--result_name "stdr_bode_icdr_deepuwf_1344_5cls" \
--fusion_layer_num 8 \
--model vit_large_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250510" \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_icdr_add_nv_prh_250509.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250510/stdr_bode_icdr_deepuwf_1344_5cls_`date +%Y%m%d_%H%M`.log 2>&1 &

# deepuwf 2 classes 0/1
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--result_name "stdr_bode_icdr_deepuwf_1344_2cls_01" \
--fusion_layer_num 8 \
--model vit_large_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250510" \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 2 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_icdr_add_nv_prh_only_01_250509.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250510/stdr_bode_icdr_deepuwf_1344_2cls_01_`date +%Y%m%d_%H%M`.log 2>&1 &

# deepuwf 2 classes dme
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--result_name "stdr_bode_icdr_deepuwf_1344_2cls_dme" \
--fusion_layer_num 8 \
--model vit_large_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250510" \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 2 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_dme_250505.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250510/stdr_bode_icdr_deepuwf_1344_2cls_dme_`date +%Y%m%d_%H%M`.log 2>&1 &

# retfound 4 classes
nohup env CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--result_name "stdr_bode_icdr_retfound_1344_4cls" \
--fusion_layer_num 8 \
--model vit_large_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(6,6), (5,5), (4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_250510" \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/RETFound_mae_natureCFP.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_bode_icdr_add_nv_prh_combine_01_250509.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode_250510/stdr_bode_icdr_retfound_1344_4cls_`date +%Y%m%d_%H%M`.log 2>&1 &
