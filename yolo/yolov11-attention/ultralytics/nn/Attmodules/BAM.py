import torch
import torch.nn as nn


class BAM(nn.Module):
    """
    BAM: Bottleneck Attention Module (BMVC 2018)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    适配说明：
      - 你的 parse_model 会自动注入 channels=C 作为第一个构造参数，
        因此 __init__ 必须以 channels 开头。

    BAM = Channel Gate + Spatial Gate，然后融合成一个注意力图 M：
      M = sigmoid( Mc + Ms )
      out = x * (1 + M)   # 论文常用 residual gating（可选）
    """

    def __init__(
        self,
        channels: int,        # C（由 parse_model 注入）
        reduction: int = 16,  # 通道门控的压缩比
        dilations=(1, 2, 4),  # 空间门控的空洞率（论文常用多尺度空洞卷积）
    ):
        super().__init__()
        assert channels > 0
        self.channels = channels

        # -------------------------
        # 1) Channel Gate (Mc)
        # -------------------------
        mid = max(1, channels // reduction)

        # GAP: (B,C,H,W)->(B,C,1,1)
        # MLP: C -> mid -> C
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                   # (B,C,1,1)
            nn.Conv2d(channels, mid, 1, bias=False),   # (B,mid,1,1)
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),   # (B,C,1,1)
        )

        # -------------------------
        # 2) Spatial Gate (Ms)
        # -------------------------
        # 先用 1x1 降维，再做多尺度空洞 3x3，最后 1x1 输出 1 通道的空间注意力
        sg_mid = max(1, channels // reduction)

        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(channels, sg_mid, 1, bias=False),  # (B,sg_mid,H,W)
            nn.BatchNorm2d(sg_mid),
            nn.ReLU(inplace=True),
        )

        # 多空洞卷积堆叠（保持 H,W 不变）
        convs = []
        for d in dilations:
            convs += [
                nn.Conv2d(sg_mid, sg_mid, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(sg_mid),
                nn.ReLU(inplace=True),
            ]
        self.spatial_convs = nn.Sequential(*convs)

        # 输出 1 通道空间门控： (B,sg_mid,H,W)->(B,1,H,W)
        self.spatial_out = nn.Conv2d(sg_mid, 1, 1, bias=False)

        # -------------------------
        # 3) Fusion
        # -------------------------
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        # Channel gate:
        # Mc: (B, C, 1, 1)
        Mc = self.channel_gate(x)

        # Spatial gate:
        # Ms: (B, 1, H, W)
        s = self.spatial_reduce(x)          # (B, sg_mid, H, W)
        s = self.spatial_convs(s)           # (B, sg_mid, H, W)
        Ms = self.spatial_out(s)            # (B, 1, H, W)

        # 融合成注意力图：
        # Mc 广播到 (B,C,H,W)，Ms 广播到 (B,C,H,W)
        # M: (B, C, H, W)
        M = self.sigmoid(Mc + Ms)

        # residual gating（论文常用）：out = x * (1 + M)
        return x * (1.0 + M)
