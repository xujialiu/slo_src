CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "octa_visionfm_224_lora8_scale2" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./OCTA_results" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--nb_classes 2 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path /data_A/xujialiu/datasets/DMI/SCP \
--csv_path /data_A/xujialiu/datasets/DMI/SCP/tabular_data/DMI_250324.csv \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]"

CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--fulltune \
--result_name "octa_visionfm_224_lora8_scale2" \
--result_root_path "./OCTA_results" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--nb_classes 2 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path /data_A/xujialiu/datasets/DMI/SCP \
--csv_path /data_A/xujialiu/datasets/DMI/SCP/tabular_data/DMI_250324.csv \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]"


nohup env CUDA_VISIBLE_DEVICES=0 python train_patches.py \
--result_name "octa_visionfm_224_lora8_scale2" \
--lora_rank 8 \
--lora_alpha 16 \
--result_root_path "./OCTA_results" \
--blr 5e-4 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 100 \
--model vit_base_patch16 \
--random_crop_perc 1 \
--finetune "/data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth" \
--data_path /data_A/xujialiu/datasets/DMI/SCP \
--csv_path /data_A/xujialiu/datasets/DMI/SCP/tabular_data/DMI_250324.csv \
--input_size "[224,224]" \
--fm_input_size 224 \
--resize_to "[224,224]" \
--batch_size 32 > ./FP_logs/q1/octa_visionfm_224_lora8_scale2_`date +%Y%m%d_%H%M`.log 2>&1 &