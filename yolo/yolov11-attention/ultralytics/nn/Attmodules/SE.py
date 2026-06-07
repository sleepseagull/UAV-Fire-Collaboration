import torch
import torch.nn as nn


class SE(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    核心步骤：
      1) Squeeze: 全局平均池化，把空间信息压缩成通道描述子 (B,C,1,1)
      2) Excitation: 两层 MLP（用 1x1 conv 等价实现）生成通道权重 (B,C,1,1)
      3) Scale: 通道权重与原特征逐元素相乘（广播到 H,W）
    """
    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: 输入特征通道数 C
            reduction: 压缩比 r，hidden = C // r（常用 16）
        """
        super().__init__()
        assert channels > 0
        hidden = max(1, channels // reduction)  # 防止 C 很小时变成 0

        # 1) Squeeze: 全局平均池化 (B,C,H,W) -> (B,C,1,1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 2) Excitation: 两层“通道 MLP”，这里用 1x1 Conv 来实现
        #    (B,C,1,1) -> (B,hidden,1,1) -> (B,C,1,1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        # 3) 将权重限制到 0~1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape  # 仅用于理解维度

        # ---- Squeeze ----
        # z: (B, C, 1, 1)
        z = self.gap(x)

        # ---- Excitation ----
        # s: (B, C, 1, 1)，每个通道一个权重，范围 0~1
        s = self.fc2(self.act(self.fc1(z)))
        s = self.sigmoid(s)

        # ---- Scale (广播乘法) ----
        # x * s: s 会在 H,W 维度上广播，所以输出仍为 (B, C, H, W)
        out = x * s

        return out


if __name__ == "__main__":
    # quick test
    x = torch.randn(2, 64, 80, 80)     # (B=2, C=64, H=80, W=80)
    se = SE(channels=64, reduction=16)
    y = se(x)
    print("x:", x.shape, "y:", y.shape)
