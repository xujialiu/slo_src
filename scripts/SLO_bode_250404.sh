nohup env CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_large_patch16_multiscales \
--result_name "SLO_bode" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/bode/tabular_data/split_250404.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32 > ./SLO_results/bode/SLO_`date +%Y%m%d_%H%M`.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--result_name "SLO_bode_visionfm" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/bode/tabular_data/split_250404.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8 > ./SLO_results/bode/SLO_visionfm_`date +%Y%m%d_%H%M`.log 2>&1 &

# fulltune 
nohup env CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--fulltune \
--use_llrd \
--layer_decay 0.6 \
--clip_grad 1 \
--result_name "SLO_bode_visionfm_fulltune" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/bode/tabular_data/split_250404.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8 > ./SLO_results/bode/SLO_bode_visionfm_fulltune_`date +%Y%m%d_%H%M`.log 2>&1 &



# fulltune testing
CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--model vit_base_patch16_multiscales \
--fulltune \
--use_llrd \
--layer_decay 0.6 \
--clip_grad 1 \
--result_name "SLO_bode_visionfm" \
--num_patches "[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/bode/tabular_data/split_250404.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--resize_to "[1344,1344]" \
--batch_size 4 \
--accum_iter 8

CUDA_VISIBLE_DEVICES=3 python train_multiscale.py \
--model vit_large_patch16_multiscales \
--fulltune \
--use_llrd \
--layer_decay 0.6 \
--clip_grad 1 \
--result_name "SLO_bode_fulltune" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/bode/tabular_data/split_250404.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32


CUDA_VISIBLE_DEVICES=0 python train_multiscale.py \
--model vit_large_patch16_multiscales \
--result_name "SLO_bode_fulltune" \
--num_patches "[(3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./SLO_results/bode" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 5 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth" \
--data_path "/data_A/xujialiu/datasets/bode/daytona_all" \
--loss_type "cross_entropy" \
--csv_path "/data_A/xujialiu/datasets/bode/tabular_data/split_250404.csv" \
--input_size "[1344,1344]" \
--fm_input_size 448 \
--resize_to "[1344,1344]" \
--batch_size 1 \
--accum_iter 32