import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from timm.models.vision_transformer import Block, PatchEmbed  # 使用 timm 库的 ViT 组件
from fmoe import FMoETransformerMLP, Top1Gate  # 使用更合适的门控

class MoEViTBlock(nn.Module):
    def __init__(self, dim, num_heads, num_experts=4, mlp_ratio=4.):
        super().__init__()
        # 保持原始注意力机制
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        
        # 将 FFN 替换为 MoE 版本
        self.norm2 = nn.LayerNorm(dim)
        self.moe_mlp = FMoETransformerMLP(
            num_expert=num_experts,
            d_model=dim,
            d_hidden=int(dim * mlp_ratio),
            gate=Top1Gate,  # 使用 Top1 门控更适合 ViT
            world_size=dist.get_world_size()  # 自动处理分布式专家分配
        )

    def forward(self, x):
        # 注意力部分保持不变
        x = x + self.attn(self.norm1(x), x, x)[0]
        
        # MoE 替换 FFN
        moe_out = self.moe_mlp(self.norm2(x))
        x = x + moe_out
        return x

class MoEViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, num_experts=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 用 MoE 块替换标准 ViT 块
        self.blocks = nn.Sequential(*[
            MoEViTBlock(embed_dim, num_heads, num_experts)
            for _ in range(depth)
        ])
        
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, C]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
            
        return self.head(x[:, 0])  # 使用分类 token

# 分布式初始化 (同之前)
dist.init_process_group(backend='nccl')
torch.cuda.set_device(dist.get_rank())

# 使用 ImageNet 示例
train_set = torchvision.datasets.ImageFolder(
    '/path/to/imagenet',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ]))
sampler = DistributedSampler(train_set)
train_loader = DataLoader(train_set, batch_size=256, sampler=sampler)

# 创建模型
model = MoEViT(
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_experts=8  # 更多专家效果更好
).cuda()
model = nn.parallel.DistributedDataParallel(model)

# 使用更大的学习率和 AdamW 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# 训练循环类似，但建议增加 epoch 数