
# step1_4layers
nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--train_step 1 \
--fusion_layer_num 4 \
--clip_grad 1 \
--result_name "2step_1_4layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_2step" \
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
--batch_size 32 > ./SLO_results/stdr_bode_2step/2step_1_4layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_`date +%Y%m%d_%H%M`.log 2>&1 &

# step1_8layers
nohup env CUDA_VISIBLE_DEVICES=1 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--train_step 1 \
--fusion_layer_num 8 \
--clip_grad 1 \
--result_name "2step_1_8layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_2step" \
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
--batch_size 32 > ./SLO_results/stdr_bode_2step/2step_1_8layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_`date +%Y%m%d_%H%M`.log 2>&1 &

# step2_4layers
nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--seed 2 \
--train_step 2 \
--fusion_layer_num 4 \
--clip_grad 1 \
--result_name "2step_2_4layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_2step" \
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
--accum_iter 8 > ./SLO_results/stdr_bode_2step/2step_2_4layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_`date +%Y%m%d_%H%M`.log 2>&1 &

# step2_8layers
nohup env CUDA_VISIBLE_DEVICES=3 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--seed 2 \
--train_step 2 \
--fusion_layer_num 8 \
--clip_grad 1 \
--result_name "2step_2_8layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_2step" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_2step/2step_1_8layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_20250412_1833/checkpoints/checkpoint_24.pth" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_bode_icdr_250410.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8 > ./SLO_results/stdr_bode_2step/2step_2_8layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1_`date +%Y%m%d_%H%M`.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python train_multiscale_2steps.py \
--model vit_base_patch16_multiscales \
--seed 2 \
--train_step 2 \
--fusion_layer_num 8 \
--clip_grad 1 \
--result_name "2step_2_8layers_stdr_bode_icdr_visionfm_lora8_scale2_rm_35_1" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/stdr_bode_2step" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "" \
--data_path "/home/xujia/datasets/uwf_dataset" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/UWF_datasets/tabular_data/stdr_rm_35_1_bode_icdr_250410.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8