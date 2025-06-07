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

    num_layers = len(model.blocks) + 1

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
        layer_id = get_layer_id_for_vit(name, num_layers)
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


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "pos_embed", "patch_pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    # head, fc_norm, fusion_block, patch_pos_embed
    elif name.startswith("head"):
        return num_layers
    else:
        return num_layers