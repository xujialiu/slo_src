import torch
import torch.nn as nn

# from timm.models.vision_transformer import VisionTransformer
from block_gradcam import TransformerBlock, trunc_normal_
import block_gradcam
from layer_gradcam import *


import torch.nn.functional as F
from einops import rearrange
from util.pos_embed import get_3d_sincos_pos_embed

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


class VisionTransformer(block_gradcam.VisionTransformer):
    def __init__(
        self,
        img_size,
        embed_dim,
        depth,
        num_classes,
        drop_path_rate=0.1,
        patch_size=16,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
    ):
        super().__init__(
            img_size=img_size,
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        # x = self.blocks(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class MultiScale(nn.Module):
    def __init__(
        self,
        input_size=(448, 448),
        num_patches=((2, 2), (1, 1)),  # for patch_model use ((4, 4))
        # fusion bock 参数
        fusion_layer_num=2,
        fusion_dropout=0.1,
        fusion_num_heads=12,
        fusion_mlp_ratio=1,
        use_learnable_pos_embed=True,
        # vit 参数
        fm_input_size=224,
        patch_size=16,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        # vit 和 fusion block 参数
        embed_dim=768,
        # head 参数
        num_classes=5,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_patches = num_patches

        self.image_encoder = VisionTransformer(
            img_size=fm_input_size,
            num_classes=0,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
        )

        total_patches = 0
        for n_h, n_w in num_patches:
            num_scale_patches = n_h * n_w
            total_patches += num_scale_patches

        self.fusion_block = FusionBlock(
            num_patches=num_patches,
            total_patches=total_patches,
            embed_dim=embed_dim,
            fusion_dropout=fusion_dropout,
            fusion_layer_num=fusion_layer_num,
            fusion_num_heads=fusion_num_heads,
            fusion_mlp_ratio=fusion_mlp_ratio,
            use_learnable_pos_embed=use_learnable_pos_embed,
        )
        self.interpolate_split_block = InterpolateSplitBlock(
            num_patches, fm_input_size, input_size
        )

        self.head = (
            Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        # B C H W -> (B n_p) C p_h p_w
        x = self.interpolate_split_block(x)
        # (B n_p) C p_h p_w -> (B n_p) embed_dim
        x = self.image_encoder(x)
        # (B n_p) embed_dim -> B n_p embed_dim

        x = rearrange(
            x,
            "(B n_p) embed_dim -> B n_p embed_dim",
            B=B,
        )
        # B n_p embed_dim -> B embed_dim
        x = self.fusion_block(x)

        # B embed_dim -> B num_classes
        x = self.head(x)
        return x

    def no_weight_decay(self):
        return {
            "image_encoder.pos_embed",  # Vision Transformer的位置嵌入
            "fusion_block.fusion_pos_embed",  # 多尺度融合模块的位置嵌入
            "image_encoder.cls_token",  # 类别标记（若存在）
        }
        
    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        # cam = self.image_encoder.pool.relprop(cam, **kwargs)
        cam = self.image_encoder.pool.relprop(cam, **kwargs)
        for blk in reversed(self.image_encoder.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.image_encoder.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam


class FusionBlock(nn.Module):
    def __init__(
        self,
        num_patches,
        total_patches,
        embed_dim,
        fusion_dropout,
        fusion_layer_num,
        fusion_num_heads=12,
        fusion_mlp_ratio=1,
        use_learnable_pos_embed=True,
    ):
        super().__init__()

        if use_learnable_pos_embed:
            self.fusion_pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    total_patches,
                    embed_dim,
                )
            )
            trunc_normal_(self.fusion_pos_embed, std=0.02)
        else:
            # (1, total_patches, embed_dim)
            pos_embed = get_3d_sincos_pos_embed(embed_dim, num_patches)
            # (1, 1, total_patches, embed_dim)
            self.register_buffer(
                "fusion_pos_embed",
                torch.tensor(pos_embed).float().unsqueeze(0),
            )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=fusion_num_heads,
                    mlp_ratio=fusion_mlp_ratio,
                    qkv_bias=True,
                    drop=fusion_dropout,
                    attn_drop=fusion_dropout,
                    drop_path=fusion_dropout,
                )
                for _ in range(fusion_layer_num)
            ]
        )
        self.fusion_norm = LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x + self.fusion_pos_embed
        # x = self.transformer_blocks(x)
        for blk in self.transformer_blocks:
            x = blk(x)
        x = self.fusion_norm(x)
        x = x.mean(dim=1)

        return x


class InterpolateSplitBlock(nn.Module):
    def __init__(self, num_patches, fm_input_size, input_size):
        super().__init__()
        self.num_patches = num_patches
        self.fm_input_size = fm_input_size
        self.input_size = input_size

    def forward(self, x):
        all_patches = []
        for n_h, n_w in self.num_patches:
            target_h = n_h * self.fm_input_size
            target_w = n_w * self.fm_input_size

            if (self.input_size[0], self.input_size[1]) != (target_h, target_w):
                scaled_x = F.interpolate(x, size=(target_h, target_w), mode="bicubic")
            else:
                scaled_x = x

            patches = self._split_to_patches(scaled_x, n_h, n_w)
            all_patches.append(patches)

        x = torch.cat(all_patches, dim=1)  # [B, Total_Patches, C, H, W]
        return rearrange(
            x,
            "B n_p C p_h p_w -> (B n_p) C p_h p_w",
        )

    def _split_to_patches(self, x, n_h, n_w):
        return rearrange(
            x,
            "B C (n_h p_h) (n_w p_w) -> B (n_h n_w) C p_h p_w",
            n_h=n_h,
            n_w=n_w,
            p_h=self.fm_input_size,
            p_w=self.fm_input_size,
        )
