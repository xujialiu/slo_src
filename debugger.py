import sys
import runpy
import os
from pathlib import Path
import shlex
import subprocess
import re


def get_cmd_list(cmd):
    def replace_backticks(match):
        command = match.group(1)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()

    cmd = re.sub(r"`([^`]+)`", replace_backticks, cmd)
    result = shlex.split(cmd)
    for idx, i in enumerate(result):
        if i == "\n":
            result.pop(idx)
    return result


def main(cmd):
    current_dir = Path(__file__).parent.resolve()
    os.chdir(current_dir)

    cmd = get_cmd_list(cmd)

    if cmd[0] == "python":
        cmd.pop(0)

    if cmd[0] == "-m":
        cmd.pop(0)
        fun = runpy.run_module
    else:
        fun = runpy.run_path

    sys.argv.extend(cmd[1:])
    fun(cmd[0], run_name="__main__")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # sample command to run the script
    cmd = """
python train_multiscale_odin.py \
--eval \
--result_root_path "./SLO_results/odin_250601" \
--result_name "other_diseases" \
--fusion_layer_num 8 \
--model vit_base_patch16_multiscales \
--dataset_type slo \
--clip_grad 1 \
--num_patches "[(6,6), (5,5), (4,4), (3,3), (2,2), (1,1)]" \
--lora_rank 8 \
--lora_alpha 16 \
--blr 1e-3 \
--layer_decay 0.65 \
--drop_path 0.1 \
--epochs 50 \
--random_crop_perc 0.9 \
--nb_classes 4 \
--data_path "/data_A/xujialiu/datasets/UWF_datasets" \
--loss_type "cross_entropy" \
--finetune "/data_A/xujialiu/projects/FMUE_SLO/SLO_results/stdr_bode_250510/stdr_bode_icdr_visionfm_1344_4cls_20250513_1814/checkpoints/checkpoint_38.pth" \
--csv_path "/data_A/xujialiu/datasets/bode/250530_get_model_tabular_data/icdr4_other_diseases_250530.csv" \
--input_size "[1344,1344]" \
--fm_input_size 224 \
--batch_size 1 \
--accum_iter 16
    """

    main(cmd)
