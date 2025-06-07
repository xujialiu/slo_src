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

