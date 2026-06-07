import torch
import torch.nn as nn


class ZPool(nn.Module):
    """
    Z-Pool: 通道维的 AvgPool + MaxPool 拼接
    输入:  (B, C, H, W)
    输出:  (B, 2, H, W)  其中 dim=1 的 2 来自 [avg, max]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # avg: (B,1,H,W), max: (B,1,H,W)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        # cat: (B,2,H,W)
        return torch.cat([avg, mx], dim=1)


class BasicConv(nn.Module):
    """
    Conv + BN (+ 可选 ReLU)
    """
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0, relu: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class AttentionGate(nn.Module):
    """
    Triplet Attention 的核心 Gate：
      x -> ZPool -> 7x7 Conv -> Sigmoid -> 与 x 相乘

    输入:  (B, C, H, W)
    输出:  (B, C, H, W)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.zpool = ZPool()
        p = kernel_size // 2
        # (B,2,H,W) -> (B,1,H,W)
        self.conv = BasicConv(2, 1, k=kernel_size, p=p, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z: (B,2,H,W)
        z = self.zpool(x)
        # attn: (B,1,H,W)
        attn = self.sigmoid(self.conv(z))
        # 广播到通道维： (B,C,H,W) * (B,1,H,W)
        return x * attn


class TripletAttention(nn.Module):
    """
    Triplet Attention (2021)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    思想：
      不只做 (C,H,W) 上的常规空间注意力，
      而是通过“维度置换”在三个视角上做注意力：
        1) (C, H) 视角：对 W 方向建模（通过 permute 实现）
        2) (C, W) 视角：对 H 方向建模
        3) (H, W) 视角：常规空间注意力
      最后把三路结果平均融合。

    适配说明：
      - 你的 parse_model 会注入 channels=C 到 __init__ 的第一个参数，
        但 TripletAttention 本身不依赖 C，所以这里 channels 仅作占位适配。
    """

    def __init__(self, channels: int, kernel_size: int = 7, no_spatial: bool = False):
        """
        Args:
            channels: 输入通道数 C（parse_model 注入，占位即可）
            kernel_size: Gate 中 7x7 卷积核大小
            no_spatial: 是否去掉常规 HW 分支（默认 False：三分支都用）
        """
        super().__init__()
        self.channels = channels
        self.no_spatial = no_spatial

        # 三个 Gate（其中 HW 可选）
        self.gate_cw = AttentionGate(kernel_size=kernel_size)  # 处理 C-W 视角（建模 H）
        self.gate_ch = AttentionGate(kernel_size=kernel_size)  # 处理 C-H 视角（建模 W）
        self.gate_hw = AttentionGate(kernel_size=kernel_size)  # 常规 H-W 视角

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        # ---- 分支1：C-W 视角（通过交换 H 和 C）----
        # x1: (B, H, C, W)
        x1 = x.permute(0, 2, 1, 3).contiguous()
        # gate 后仍是 (B, H, C, W)
        x1 = self.gate_cw(x1)
        # 变回 (B, C, H, W)
        x1 = x1.permute(0, 2, 1, 3).contiguous()

        # ---- 分支2：C-H 视角（通过交换 W 和 C）----
        # x2: (B, W, H, C)
        x2 = x.permute(0, 3, 2, 1).contiguous()
        x2 = self.gate_ch(x2)
        # 变回 (B, C, H, W)
        x2 = x2.permute(0, 3, 2, 1).contiguous()

        # ---- 分支3：常规 HW 视角 ----
        if self.no_spatial:
            out = (x1 + x2) / 2.0
        else:
            x3 = self.gate_hw(x)  # (B,C,H,W)
            out = (x1 + x2 + x3) / 3.0

        return out
