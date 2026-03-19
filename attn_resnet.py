# Kimi Team et al., "Attention Residuals," arXiv, 2026, https://arxiv.org/abs/2603.15031.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class RMSNorm(nn.Module):
    """RMSNorm层，用于AttnRes的注意力权重计算"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        return x * self.weight / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class AttnResBlock(nn.Module):
    """带AttnRes的ResNet基础块（支持BasicBlock/Bottleneck）"""
    expansion: int = 1

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        downsample: nn.Module = None,
        dim: int = 512,  # AttnRes的特征维度
        eps: float = 1e-6
    ):
        super().__init__()
        # 标准ResNet卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

        # AttnRes相关参数
        self.dim = dim
        self.rms_norm = RMSNorm(dim, eps)
        # 可学习的伪查询向量（对应论文中的w_l）
        self.query = nn.Parameter(torch.zeros(dim))
        # 初始化
        nn.init.zeros_(self.query)

    def forward(self, x: torch.Tensor, prev_block_reprs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 当前块输入 (B, C, H, W)
            prev_block_reprs: 前序块的表示列表 [(B, dim), ...]
        Returns:
            out: 当前块输出 (B, C, H, W)
            block_repr: 当前块的表示 (B, dim)
        """
        # 标准ResNet前向
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 计算当前块的全局表示（池化+投影到dim维度）
        b, c, h, w = out.shape
        # 全局平均池化: (B, C)
        global_feat = F.adaptive_avg_pool2d(out, (1, 1)).view(b, c)
        # 投影到AttnRes的dim维度
        if c != self.dim:
            proj = nn.Linear(c, self.dim, device=x.device, dtype=x.dtype)
            block_repr = proj(global_feat)
        else:
            block_repr = global_feat

        # AttnRes注意力聚合（仅训练阶段启用，推理可简化）
        if self.training and len(prev_block_reprs) > 0:
            # 堆叠前序块表示: (num_prev, B, dim)
            prev_reprs = torch.stack(prev_block_reprs, dim=0)
            # RMSNorm归一化
            prev_reprs_norm = self.rms_norm(prev_reprs)
            # 计算注意力权重: (B, num_prev)
            attn_scores = torch.einsum('d, n b d -> b n', self.query, prev_reprs_norm)
            attn_weights = F.softmax(attn_scores / self.dim**0.5, dim=-1)
            # 聚合前序信息: (B, dim)
            aggregated = torch.einsum('b n, n b d -> b d', attn_weights, prev_reprs)
            # 投影回特征维度并加到identity上
            if self.dim != c:
                proj_back = nn.Linear(self.dim, c, device=x.device, dtype=x.dtype)
                aggregated_feat = proj_back(aggregated).view(b, c, 1, 1)
            else:
                aggregated_feat = aggregated.view(b, c, 1, 1)
            identity = identity + aggregated_feat

        # 最终输出
        out += identity
        out = self.relu(out)
        return out, block_repr


class BottleneckAttnResBlock(AttnResBlock):
    """带AttnRes的Bottleneck块（用于ResNet50/101/152）"""
    expansion = 4

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        downsample: nn.Module = None,
        dim: int = 512,
        eps: float = 1e-6
    ):
        super().__init__(in_channels, out_channels, stride, downsample, dim, eps)
        # 替换为Bottleneck卷积结构
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)


class AttnResNet(nn.Module):
    """集成Block AttnRes的ResNet完整实现"""
    def __init__(
        self, 
        block: nn.Module,
        layers: List[int],
        num_classes: int = 1000,
        attn_res_dim: int = 512,
        num_attn_blocks: int = 8,  # 对应论文中的N=8
        eps: float = 1e-6
    ):
        super().__init__()
        self.in_channels = 64
        self.attn_res_dim = attn_res_dim
        self.num_attn_blocks = num_attn_blocks
        self.eps = eps

        # 初始卷积层（标准ResNet）
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建带AttnRes的layer
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, 
        block: nn.Module, 
        out_channels: int, 
        blocks: int, 
        stride: int = 1
    ) -> nn.Sequential:
        """构建AttnRes层"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个块（带downsample）
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.attn_res_dim, self.eps))
        self.in_channels = out_channels * block.expansion
        # 剩余块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dim=self.attn_res_dim, eps=self.eps))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（集成Block AttnRes）"""
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 维护块级表示列表
        block_reprs = []
        # Layer 1
        for block in self.layer1:
            x, repr = block(x, block_reprs)
            block_reprs.append(repr)
            # 控制块数量（Block AttnRes）
            if len(block_reprs) > self.num_attn_blocks:
                block_reprs = block_reprs[-self.num_attn_blocks:]
        
        # Layer 2
        for block in self.layer2:
            x, repr = block(x, block_reprs)
            block_reprs.append(repr)
            if len(block_reprs) > self.num_attn_blocks:
                block_reprs = block_reprs[-self.num_attn_blocks:]
        
        # Layer 3
        for block in self.layer3:
            x, repr = block(x, block_reprs)
            block_reprs.append(repr)
            if len(block_reprs) > self.num_attn_blocks:
                block_reprs = block_reprs[-self.num_attn_blocks:]
        
        # Layer 4
        for block in self.layer4:
            x, repr = block(x, block_reprs)
            block_reprs.append(repr)
            if len(block_reprs) > self.num_attn_blocks:
                block_reprs = block_reprs[-self.num_attn_blocks:]

        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 预定义AttnResNet模型（对应不同规模的ResNet）
def attn_resnet18(num_classes: int = 1000, attn_res_dim: int = 512, num_attn_blocks: int = 8) -> AttnResNet:
    return AttnResNet(AttnResBlock, [2, 2, 2, 2], num_classes, attn_res_dim, num_attn_blocks)

def attn_resnet50(num_classes: int = 1000, attn_res_dim: int = 512, num_attn_blocks: int = 8) -> AttnResNet:
    return AttnResNet(BottleneckAttnResBlock, [3, 4, 6, 3], num_classes, attn_res_dim, num_attn_blocks)

def attn_resnet101(num_classes: int = 1000, attn_res_dim: int = 512, num_attn_blocks: int = 8) -> AttnResNet:
    return AttnResNet(BottleneckAttnResBlock, [3, 4, 23, 3], num_classes, attn_res_dim, num_attn_blocks)


# 测试代码
if __name__ == "__main__":
    # 初始化模型
    model = attn_resnet18(num_classes=1000, attn_res_dim=512, num_attn_blocks=8)
    model.eval()  # 推理模式

    # 测试输入（batch_size=2, 3通道, 224x224）
    dummy_input = torch.randn(2, 3, 224, 224)
    # 前向传播
    output = model(dummy_input)
    
    print(f"模型输出形状: {output.shape}")  # 应输出 (2, 1000)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
