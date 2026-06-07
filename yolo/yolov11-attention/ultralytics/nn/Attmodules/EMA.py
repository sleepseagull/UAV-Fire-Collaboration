import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention (EMA)

    核心：
    - 通道分组：把 (B, C, H, W) reshape 成 (B*g, C/g, H, W)，等价“把组当作 batch 扩展”
    - 两个并行分支：
        (1) 1x1 + (沿H/W编码的门控)  -> x1
        (2) 3x3 局部交互            -> x3
    - Cross-spatial learning：
        用 softmax 在 HW 上形成“空间分布”，再用 bmm 生成 1xHxW 的空间注意力图
    - 输出：xg * s -> reshape 回 (B, C, H, W)
    """

    def __init__(self, channels: int, groups: int = 32, act: Optional[nn.Module] = None):
        super().__init__()
        assert channels > 0, "channels must be positive"
        self.channels = channels

        # 让 groups 合法：<= channels 且能整除 channels
        g = min(groups, channels)
        while g > 1 and channels % g != 0:
            g -= 1
        self.groups = max(g, 1)
        self.cg = channels // self.groups  # channels per group

        self.act = act if act is not None else nn.SiLU(inplace=True)

        # 1x1 分支（CA-shared component）
        self.conv1 = nn.Conv2d(self.cg, self.cg, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_h = nn.Conv2d(self.cg, self.cg, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(self.cg, self.cg, kernel_size=1, stride=1, padding=0, bias=True)

        # 3x3 分支（局部交互）
        self.conv3 = nn.Conv2d(self.cg, self.cg, kernel_size=3, stride=1, padding=1, bias=True)

        # 论文图里在 group 空间做归一化
        self.gn = nn.GroupNorm(num_groups=1, num_channels=self.cg)

        self.sigmoid = nn.Sigmoid()

        # ✅ 修复点：不要用 lambda（会导致 torch.save / pickle 失败）
        # softmax over spatial positions (HW) to form 2D distribution for each channel
        self.softmax_spatial = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        b, c, h, w = x.shape
        assert c == self.channels, f"EMA expected {self.channels} channels, got {c}"

        g = self.groups
        cg = self.cg
        hw = h * w

        # ---- 分组： (B, C, H, W) -> (B*g, C/g, H, W)
        xg = x.view(b, g, cg, h, w).reshape(b * g, cg, h, w)

        # =========================
        # Branch A: 1x1 分支（沿 H/W 编码的门控）
        # =========================
        # 沿 W 平均 -> (B*g, cg, H, 1)
        x_h = xg.mean(dim=3, keepdim=True)

        # 沿 H 平均 -> (B*g, cg, 1, W) -> 变成 (B*g, cg, W, 1) 便于 concat
        x_w = xg.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        # concat 到 “H+W” 维度上
        y = torch.cat([x_h, x_w], dim=2)  # (B*g, cg, H+W, 1)
        y = self.act(self.conv1(y))

        # split 回 H 和 W
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # (B*g, cg, 1, W)

        # 门控权重
        a_h = self.sigmoid(self.conv_h(y_h))  # (B*g, cg, H, 1)
        a_w = self.sigmoid(self.conv_w(y_w))  # (B*g, cg, 1, W)

        # 加权得到 x1
        x1 = xg * a_h * a_w  # (B*g, cg, H, W)
        x1 = self.gn(x1)

        # =========================
        # Branch B: 3x3 分支（局部交互）
        # =========================
        x3 = self.act(self.conv3(xg))  # (B*g, cg, H, W)

        # =========================
        # Cross-spatial learning (matmul-based)
        # =========================
        # 把 (H,W) 展平到 HW 维度，对每个通道在 HW 上做 softmax -> “空间分布”
        A1 = self.softmax_spatial(x1.view(b * g, cg, hw))  # (B*g, cg, HW)
        A3 = self.softmax_spatial(x3.view(b * g, cg, hw))  # (B*g, cg, HW)

        # 全局向量（GAP）：(B*g, 1, cg)
        v1 = x1.mean(dim=(2, 3)).unsqueeze(1)
        v3 = x3.mean(dim=(2, 3)).unsqueeze(1)

        # bmm： (B*g, 1, cg) @ (B*g, cg, HW) -> (B*g, 1, HW) -> reshape (B*g,1,H,W)
        s1 = torch.bmm(v1, A3).view(b * g, 1, h, w)  # 1x1(global) x 3x3(spatial)
        s2 = torch.bmm(v3, A1).view(b * g, 1, h, w)  # 3x3(global) x 1x1(spatial)

        # 最终空间注意力图
        s = self.sigmoid(s1 + s2)  # (B*g, 1, H, W)

        out_g = xg * s  # (B*g, cg, H, W)

        # ---- 恢复维度： (B*g, C/g, H, W) -> (B, C, H, W)
        out = out_g.view(b, g, cg, h, w).reshape(b, c, h, w)
        return out
