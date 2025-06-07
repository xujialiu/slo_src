# fulltune visionfm
nohup env CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--fulltune \
--use_llrd \
--layer_decay 0.6 \
--clip_grad 1 \
--result_name "stdr_bode_icdr_visionfm_fulltune" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/home/xujia/datasets/uwf_dataset/tabular_data/stdr_bode_icdr_250407.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8 > ./SLO_results/stdr_bode/stdr_bode_icdr_visionfm_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# fulltune deepuwf
nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_large_patch16_multiscales \
--fulltune \
--use_llrd \
--layer_decay 0.6 \
--clip_grad 1 \
--result_name "stdr_bode_icdr_deepuwf_fulltune" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/home/xujia/datasets/uwf_dataset/tabular_data/stdr_bode_icdr_250407.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora visionfm
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--clip_grad 1 \
--result_name "stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/home/xujia/datasets/uwf_dataset/tabular_data/stdr_bode_icdr_250407.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8 > ./SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &

# lora visionfm rm 35_1
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--clip_grad 1 \
--result_name "stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_bode_icdr_250410.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8 > ./SLO_results/stdr_bode/stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_`date +%Y%m%d_%H%M`.log 2>&1 &

# eval lora visionfm rm 35_1
CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--eval \
--clip_grad 1 \
--result_name "eval_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_20250410_1540/checkpoints/checkpoint_14.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_bode_icdr_250410.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8

CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--eval \
--clip_grad 1 \
--result_name "eval_34_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_20250410_1540/checkpoints/checkpoint_34.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_bode_icdr_250410.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8

CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--eval \
--clip_grad 1 \
--result_name "eval_14_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_enhanced" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_20250410_1540/checkpoints/checkpoint_14.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_datasets_enhanced" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_bode_icdr_250410.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8

# lora visionfm train_all
nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--dataset_type slo \
--train_all \
--clip_grad 1 \
--result_name "stdr_bode_icdr_visionfm_lora8_scale2_trainall" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/home/xujia/datasets/uwf_dataset/tabular_data/stdr_bode_icdr_250407.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 2 \
--accum_iter 16 > ./SLO_results/stdr_bode/stdr_bode_icdr_visionfm_lora8_scale2_trainall_`date +%Y%m%d_%H%M`.log 2>&1 &


# lora visionfm eval
CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--eval \
--clip_grad 1 \
--result_name "eval_stdr_bode_icdr_visionfm_lora8_scale2" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_visionfm_lora8_scale2_20250407_1521/checkpoints/checkpoint_18.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/home/xujia/datasets/uwf_dataset/tabular_data/stdr_bode_icdr_250407.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8


# lora deepuwf
nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_large_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--result_name "stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_bode_icdr_250410.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_`date +%Y%m%d_%H%M`.log 2>&1 &

# eval lora deepuwf
CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--model vit_large_patch16_multiscales \
--eval \
--clip_grad 1 \
--result_name "eval_34_stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_20250414_1007/checkpoints/checkpoint_34.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_bode_icdr_250410.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32

# eval Tsukazaki
CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--eval \
--model vit_large_patch16_multiscales \
--clip_grad 1 \
--result_name "eval_Tsukazaki_pseudo_label" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_20250414_1007/checkpoints/checkpoint_34.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/Tsukazaki" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/Tsukazaki_pseudo_label_250416.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 

# eval GDPH_UWF
CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--eval \
--model vit_large_patch16_multiscales \
--clip_grad 1 \
--result_name "eval_GDPH_UWF_pseudo_label" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_20250414_1007/checkpoints/checkpoint_34.pth" \
--data_path "/data_A/xujialiu/datasets/UWF_external/GDPH_UWF/GDPH_UWF" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/dataset_tabular/GDPH_UWF_pseudo_label_250418.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 

# eval stdr_35
CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--eval \
--dataset_type slo \
--model vit_large_patch16_multiscales \
--clip_grad 1 \
--result_name "eval_001-327_etdr_35" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_20250414_1007/checkpoints/checkpoint_34.pth" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/tabular_data/001-327_etdr_35.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--batch_size 1 \
--accum_iter 32 

CUDA_VISIBLE_DEVICES=2 python train_multiscale.py \
--eval \
--dataset_type slo \
--model vit_large_patch16_multiscales \
--clip_grad 1 \
--result_name "eval_328-700_etdr_35" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_20250414_1007/checkpoints/checkpoint_34.pth" \
--data_path "/data_A/xujialiu/datasets/daytona/daytona" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/projects/FMUE_SLO/tabular_data/328-700_etdr_35.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--batch_size 1 \
--accum_iter 32 

# combined_12
nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_large_patch16_multiscales \
--clip_grad 1 \
--result_name "stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_combined_12" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_combined_12_bode_icdr_250417.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_combined_12_`date +%Y%m%d_%H%M`.log 2>&1 &

# combined_12_resume
nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_large_patch16_multiscales \
--clip_grad 1 \
--result_name "stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_combined_12_resume" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--resume "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_combined_12_20250417_1313/checkpoints/checkpoint_17.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_combined_12_bode_icdr_250417.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/stdr_bode/stdr_bode_icdr_deepuwf_lora8_scale2_rm_35_1_combined_12_resume_`date +%Y%m%d_%H%M`.log 2>&1 &