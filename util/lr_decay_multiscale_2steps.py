import json


def param_groups_lrd(
    model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    image_encoder_layer_num = len(model.image_encoder.blocks)
    fusion_block_layer_num = len(model.fusion_block.transformer_blocks)

    num_layers = image_encoder_layer_num + fusion_block_layer_num + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for name, params in model.named_parameters():
        if not params.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if params.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        # layer_id is the num of blocks close to the input
        layer_id = get_layer_id_for_vit(
            name,
            image_encoder_layer_num,
            fusion_block_layer_num,
        )
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(name)
        param_groups[group_name]["params"].append(params)

    sorted_param_group_names = {
        key: param_group_names[key]
        for key in sorted(param_group_names.keys(), key=lambda x: int(x.split("_")[1]))
    }
    print(f"parameter groups: \n{json.dumps(sorted_param_group_names, indent=2)}")

    return list(param_groups.values())


def get_layer_id_for_vit(
    name,
    image_encoder_layer_num,
    fusion_block_layer_num,
):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["image_encoder.cls_token", "image_encoder.pos_embed"]:
        return 0
    elif name.startswith("image_encoder.patch_embed"):
        return 0
    elif name.startswith("image_encoder.blocks"):
        return int(name.split(".")[2]) + 1
    # norm, fusion_block, fusion_pos_embed, head
    elif name.startswith("fusion_block.transformer_blocks"):
        return int(name.split(".")[2]) + image_encoder_layer_num + 1
    # norm, fusion_block, fusion_pos_embed, head
    elif name.startswith("head"):
        return image_encoder_layer_num + fusion_block_layer_num + 1
