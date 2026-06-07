import torch
import torch.nn as nn
import torch.nn.functional as F


class A2(nn.Module):
    """
    A2-Nets: Double Attention Block (NeurIPS 2018)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    适配说明：
      - 你的 parse_model 会把 channels=C 自动注入到构造函数第一个参数，
        所以这里 __init__(channels, ...) 必须以 channels 开头。

    核心思想：Gather + Distribute（双注意力）
      1) Gather：从全局空间 (H*W) 里“聚合”出一组全局特征描述子
      2) Distribute：再把这些全局描述子按位置需要“分配”回每个位置
    """

    def __init__(self, channels: int, reduction: int = 4, L: int = 32):
        """
        Args:
            channels: 输入通道数 C（由 parse_model 注入）
            reduction: 压缩比，inter = max(C//reduction, L)
            L: inter 的最小值（防止通道太小）
        """
        super().__init__()
        assert channels > 0
        self.channels = channels
        inter = max(channels // reduction, L)  # 中间维度 d
        self.inter = inter

        # 3 个 1x1 conv：分别产生 A（gather 权重）、B（被聚合的特征）、V（distribute 权重）
        self.convA = nn.Conv2d(channels, inter, kernel_size=1, bias=False)  # -> (B, inter, H, W)
        self.convB = nn.Conv2d(channels, inter, kernel_size=1, bias=False)  # -> (B, inter, H, W)
        self.convV = nn.Conv2d(channels, inter, kernel_size=1, bias=False)  # -> (B, inter, H, W)

        # 输出投影回 C
        self.proj = nn.Sequential(
            nn.Conv2d(inter, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # -------- Step 1) 生成三路特征 --------
        # A_map: (B, inter, H, W) -> (B, inter, N)
        A_map = self.convA(x).view(B, self.inter, N)
        # B_feat: (B, inter, H, W) -> (B, inter, N)
        B_feat = self.convB(x).view(B, self.inter, N)
        # V_map: (B, inter, H, W) -> (B, inter, N)
        V_map = self.convV(x).view(B, self.inter, N)

        # -------- Step 2) Gather：在空间维 N 上 softmax，做“全局聚合” --------
        # A_attn: (B, inter, N)，对每个“通道分组/基向量”在空间上归一化
        A_attn = F.softmax(A_map, dim=2)

        # G: (B, inter, inter)
        # (B, inter, N) @ (B, N, inter) -> (B, inter, inter)
        G = torch.bmm(B_feat, A_attn.permute(0, 2, 1))

        # -------- Step 3) Distribute：在 inter 维上 softmax，做“按位置分配” --------
        # V_attn: (B, inter, N)，对每个位置在 inter 维归一化（分配权重）
        V_attn = F.softmax(V_map, dim=1)

        # Y: (B, inter, N)
        # (B, inter, inter) @ (B, inter, N) -> (B, inter, N)
        Y = torch.bmm(G, V_attn)

        # (B, inter, N) -> (B, inter, H, W)
        Y = Y.view(B, self.inter, H, W)

        # -------- Step 4) 投影回 C + 残差 --------
        out = self.proj(Y)  # (B, C, H, W)
        return x + out
