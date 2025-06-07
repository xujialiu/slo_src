from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer

# from timm.models.vision_transformer import Block
from block import TransformerBlock

from timm.models.layers.weight_init import trunc_normal_
import torch
import torch.nn.functional as F


class VisionTransformerMultiScale(timm.models.vision_transformer.VisionTransformer):
    def __init__(
        self,
        norm_layer,
        embed_dim,
        input_size,
        num_patches=[(4, 4), (3, 3), (2, 2), (1, 1)],
        fm_input_size=448,
        fusion_layer_num=2,
        fusion_dropout=0.1,
        **kwargs,
    ):
        super().__init__(
            norm_layer=norm_layer, embed_dim=embed_dim, img_size=fm_input_size, **kwargs
        )
        self.input_size = input_size
        self.fc_norm = norm_layer(embed_dim)
        del self.norm

        total_patches = 0
        for n_h, n_w in num_patches:
            num_scale_patches = n_h * n_w
            total_patches += num_scale_patches

        self.fusion_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                total_patches,
                embed_dim,
            )
        )
        self.patch_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                total_patches,
                self.patch_embed.num_patches + self.num_tokens,
                embed_dim,
            )
        )
        trunc_normal_(self.patch_pos_embed, std=0.02)

        # Feature fusion transformer
        self.fusion_block = nn.Sequential(
            *[
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=16,
                    mlp_ratio=1,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop=fusion_dropout,
                    attn_drop=fusion_dropout,
                    drop_path=fusion_dropout,
                )
                for _ in range(fusion_layer_num)
            ]
        )

        self.num_patches = num_patches
        self.fm_input_size = fm_input_size
        self.total_patches = total_patches

    def _split_to_patches(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Split resized image into target patches"""
        B, C, H, W = x.shape
        ph, pw = H // n_h, W // n_w

        x = x.view(B, C, n_h, ph, n_w, pw)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, n_h * n_w, C, ph, pw)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        all_patches = []

        # 生成多尺度特征并合并
        for n_h, n_w in self.num_patches:
            # 调整输入尺寸保证可分割
            target_h = n_h * self.fm_input_size
            target_w = n_w * self.fm_input_size

            if (self.input_size[0] == target_h) and (self.input_size[1] == target_w):
                scaled_x = x
            else:
                scaled_x = F.interpolate(x, size=(target_h, target_w), mode="bicubic")

            # 分割特征图
            patches = self._split_to_patches(
                scaled_x, n_h, n_w
            )  # [B, n_h*n_w, C, H, W]

            # 合并到总特征
            all_patches.append(patches)

        # 统一处理所有patches
        all_patches = torch.cat(all_patches, dim=1)  # [B, Total_Patches, C, H, W]

        B, TP, C, H, W = all_patches.shape
        x = all_patches.view(B * TP, C, H, W)  # [B, Total_Patches, C, H, W]

        # Patch embedding
        x = self.patch_embed(x)  # [B*TP, P, D]

        # 添加class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B*TP, P+1, D]
        B_TP, P_1, D = x.shape

        # 位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)  # [B*TP, P+1, D]

        # patch 位置编码
        x = x.view(B, TP, P_1, D)  # [B, TP, P+1, D]
        x = x + self.patch_pos_embed  # [B, TP, P+1, D]
        x = x.view(B * TP, P_1, D)  # [B*TP, P+1, D]

        # Transformer处理
        x = self.blocks(x)  # [B*TP, P+1, D]  [B, seq_len, embed_dim]

        # 重构特征维度
        # x = x[:, 1:, :].mean(dim=1)  # [B*TP, D]
        x = x.mean(dim=1)
        x = x.view(B, self.total_patches, -1)  # [B, TP, D]
        x = x + self.fusion_pos_embed  # [B, TP, D]

        # 特征融合
        x = self.fusion_block(x)  # [B, TP, D]

        # 分类处理
        x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head(x)

        return x


class VisionTransformerMultiScaleWithoutPPE(
    timm.models.vision_transformer.VisionTransformer
):
    """Multi-scale ViT with unified feature processing"""

    def __init__(
        self,
        norm_layer,
        embed_dim,
        input_size,
        num_patches=[(4, 4), (3, 3), (2, 2), (1, 1)],
        fm_input_size=448,
        fusion_layer_num=2,
        fusion_dropout=0.1,
        **kwargs,
    ):
        super().__init__(
            norm_layer=norm_layer, embed_dim=embed_dim, img_size=fm_input_size, **kwargs
        )
        self.input_size = input_size
        self.fc_norm = norm_layer(embed_dim)
        del self.norm

        total_patches = 0
        for n_h, n_w in num_patches:
            num_scale_patches = n_h * n_w
            total_patches += num_scale_patches

        self.fusion_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                total_patches,
                embed_dim,
            )
        )

        # Feature fusion transformer
        self.fusion_block = nn.Sequential(
            *[
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=16,
                    mlp_ratio=1,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop=fusion_dropout,
                    attn_drop=fusion_dropout,
                    drop_path=fusion_dropout,
                )
                for _ in range(fusion_layer_num)
            ]
        )

        self.num_patches = num_patches
        self.fm_input_size = fm_input_size
        self.total_patches = total_patches

    def _split_to_patches(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Split resized image into target patches"""
        B, C, H, W = x.shape
        ph, pw = H // n_h, W // n_w

        x = x.view(B, C, n_h, ph, n_w, pw)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, n_h * n_w, C, ph, pw)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        all_patches = []

        # 生成多尺度特征并合并
        for n_h, n_w in self.num_patches:
            # 调整输入尺寸保证可分割
            target_h = n_h * self.fm_input_size
            target_w = n_w * self.fm_input_size

            if (self.input_size[0] == target_h) and (self.input_size[1] == target_w):
                scaled_x = x
            else:
                scaled_x = F.interpolate(x, size=(target_h, target_w), mode="bicubic")

            # 分割特征图
            patches = self._split_to_patches(
                scaled_x, n_h, n_w
            )  # [B, n_h*n_w, C, H, W]

            # 合并到总特征
            all_patches.append(patches)

        # 统一处理所有patches
        all_patches = torch.cat(all_patches, dim=1)  # [B, Total_Patches, C, H, W]
        # all_patches = all_patches + self.patch_pos_embed  # [B, Total_Patches, C, H, W]

        B, TP, C, H, W = all_patches.shape
        x = all_patches.view(B * TP, C, H, W)  # [B, Total_Patches, C, H, W]

        # Patch embedding
        x = self.patch_embed(x)  # [B*TP, P, D]

        # 添加class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B*TP, P+1, D]

        # 位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)  # [B*TP, P+1, D]

        # Transformer处理
        x = self.blocks(x)  # [B*TP, P+1, D]

        # 重构特征维度
        x = x[:, 1:, :].mean(dim=1)  # [B*TP, D]
        x = x.view(B, self.total_patches, -1)  # [B, TP, D]
        x = x + self.fusion_pos_embed

        # 特征融合
        x = self.fusion_block(x)

        # 分类处理
        x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head(x)

        return x


class VisionTransformerPatches(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with patch-level position encoding and feature fusion"""

    def __init__(
        self,
        norm_layer,
        embed_dim,
        input_size,
        fm_input_size=448,
        fusion_layer_num=2,
        fusion_dropout=0.1,
        **kwargs,
    ):
        super(VisionTransformerPatches, self).__init__(
            norm_layer=norm_layer, embed_dim=embed_dim, img_size=fm_input_size, **kwargs
        )
        self.fm_input_size = fm_input_size
        self.fc_norm = norm_layer(embed_dim)
        del self.norm

        h, w = input_size

        num_patches = (h // fm_input_size) * (w // fm_input_size)

        if h % fm_input_size != 0 or w % fm_input_size != 0:
            raise ValueError("Image dimensions must be divisible by fm_input_size")

        self.patch_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches,
                self.patch_embed.num_patches + self.num_tokens,
                embed_dim,
            )
        )
        self.fusion_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.patch_pos_embed, std=0.02)
        trunc_normal_(self.fusion_pos_embed, std=0.02)

        # 新增用于融合patch特征的transformer层
        self.fusion_block = nn.Sequential(
            *[
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=16,  # 与原始头数保持一致
                    mlp_ratio=1,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop=fusion_dropout,
                    attn_drop=fusion_dropout,
                    drop_path=fusion_dropout,
                )
                for _ in range(fusion_layer_num)  # 使用2层transformer
            ]
        )

    def forward_features(self, x):
        # (B, n_h*n_w, C, patch_size, patch_size)
        x = self._split_to_patches(x, patch_size=self.fm_input_size)

        b, n, c, h, w = x.shape

        x = x.reshape(b * n, c, h, w)

        # Pass through patch embedding
        x = self.patch_embed(
            x
        )  # (B*N, P, D) where P is num patches per image, D is embed_dim

        # Expand cls token for each patch
        cls_tokens = self.cls_token.expand(b * n, -1, -1)  # (B*N, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B*N, P+1, D)

        # Add position embeddings
        x = x + self.pos_embed  # (B*N, P+1, D)
        x = self.pos_drop(x)

        B_N, P_1, D = x.shape

        x = x.view(b, -1, P_1, D)
        # breakpoint()
        x = x + self.patch_pos_embed  # (B, N, P+1, D)
        
        x = x.view(B_N, P_1, D)

        # Pass through transformer blocks
        x = self.blocks(x)

        # Reshape back to (B, N, P+1, D)
        x = x.view(b, n, P_1, D)

        return x

    def forward(self, x):
        # 特征提取
        x = self.forward_features(x)

        x = x[:, :, 1:, :].mean(dim=2)

        # 添加patch位置编码
        x += self.fusion_pos_embed

        # 通过transformer层进行特征交互
        x = self.fusion_block(x)

        # 全局平均池化
        x = x.mean(dim=1)

        # 分类头处理
        x = self.fc_norm(x)
        x = self.head(x)

        return x

    def _split_to_patches(
        self, image: torch.Tensor, patch_size: int = 448
    ) -> torch.Tensor:

        assert image.dim() == 4, f"Expected 4D input tensor, got {image.dim()}D"
        B, C, H, W = image.shape
        assert (
            H % patch_size == 0 and W % patch_size == 0
        ), f"Image dimensions ({H}, {W}) must be divisible by patch_size {patch_size}"

        n_h = H // patch_size
        n_w = W // patch_size

        # (B, C, H, W) -> (B, C, n_h, patch_size, W)
        x = image.view(B, C, n_h, patch_size, W)

        # (B, C, n_h, patch_size, W) -> (B, C, n_h, patch_size, n_w, patch_size)
        x = x.view(B, C, n_h, patch_size, n_w, patch_size)

        # (B, C, n_h, patch_size, n_w, patch_size) -> (B, n_h*n_w, C, patch_size, patch_size)
        patches = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, n_h * n_w, C, patch_size, patch_size)

        return patches


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, norm_layer, embed_dim, **kwargs):
        super(VisionTransformer, self).__init__(
            norm_layer=norm_layer, embed_dim=embed_dim, **kwargs
        )

        self.fc_norm = norm_layer(embed_dim)
        del self.norm  # remove the original norm

    def forward_features(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        outcome = self.fc_norm(x)

        return outcome


def vit_large_patch16(input_size, num_classes, drop_path_rate):
    if str(type(input_size)) != "<class 'int'>":
        input_size = input_size[0]
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=input_size,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model


def vit_base_patch16(input_size, num_classes, drop_path_rate):
    if str(type(input_size)) != "<class 'int'>":
        input_size = input_size[0]
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=input_size,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model


def vit_large_patch16_patches(
    input_size,
    num_classes,
    drop_path_rate,
    fusion_layer_num,
    fusion_dropout,
    fm_input_size=448,
):
    model = VisionTransformerPatches(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        input_size=input_size,  # 真正的图像输入大小
        fm_input_size=fm_input_size,  # 用于特征提取的网络的输入大小
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        fusion_layer_num=fusion_layer_num,
        fusion_dropout=fusion_dropout,
    )
    return model


def vit_base_patch16_patches(
    input_size,
    num_classes,
    drop_path_rate,
    fusion_layer_num,
    fusion_dropout,
    fm_input_size=448,
):
    model = VisionTransformerPatches(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        input_size=input_size,  # 真正的图像输入大小
        fm_input_size=fm_input_size,  # 用于特征提取的网络的输入大小
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        fusion_layer_num=fusion_layer_num,
        fusion_dropout=fusion_dropout,
    )
    return model


def vit_base_patch16_multiscales(
    input_size,
    num_classes,
    drop_path_rate,
    fusion_layer_num,
    fusion_dropout,
    fm_input_size=448,
    num_patches=[(4, 4), (2, 2), (1, 1)],
):
    model = VisionTransformerMultiScale(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        input_size=input_size,  # 真正的图像输入大小
        fm_input_size=fm_input_size,  # 用于特征提取的网络的输入大小
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        fusion_layer_num=fusion_layer_num,
        fusion_dropout=fusion_dropout,
        num_patches=num_patches,
    )
    return model


def vit_large_patch16_multiscales(
    input_size,
    num_classes,
    drop_path_rate,
    fusion_layer_num,
    fusion_dropout,
    fm_input_size=448,
    num_patches=[(4, 4), (2, 2), (1, 1)],
):
    model = VisionTransformerMultiScale(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        input_size=input_size,  # 真正的图像输入大小
        fm_input_size=fm_input_size,  # 用于特征提取的网络的输入大小
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        fusion_layer_num=fusion_layer_num,
        fusion_dropout=fusion_dropout,
        num_patches=num_patches,
    )
    return model


def vit_base_patch16_multiscales_without_ppe(
    input_size,
    num_classes,
    drop_path_rate,
    fusion_layer_num,
    fusion_dropout,
    fm_input_size=448,
    num_patches=[(4, 4), (2, 2), (1, 1)],
):
    model = VisionTransformerMultiScaleWithoutPPE(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        input_size=input_size,  # 真正的图像输入大小
        fm_input_size=fm_input_size,  # 用于特征提取的网络的输入大小
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        fusion_layer_num=fusion_layer_num,
        fusion_dropout=fusion_dropout,
        num_patches=num_patches,
    )
    return model
