import torch
import torch.nn as nn


class CA(nn.Module):
    """
    Coordinate Attention (CA)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    核心思想：
      - 分别在 H 方向、W 方向做全局池化，得到两条“方向感知”的通道描述
      - 共享 1x1 变换后再分成两支，生成 Ah(B,C,H,1) 与 Aw(B,C,1,W)
      - 最终用 Ah 与 Aw 对输入逐元素重标定
    """

    def __init__(self, channels: int, reduction: int = 32):
        """
        Args:
            channels: 输入通道数 C
            reduction: 压缩比，论文常用 32（也可用 16）
        """
        super().__init__()
        assert channels > 0
        mip = max(8, channels // reduction)  # 论文里常设最小 8

        # 共享变换：C -> mip (1x1 conv)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        # 两个分支：mip -> C
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape

        # ---- Step 1: 分方向池化（保留坐标信息）----
        # x_h: (B, C, H, 1)  沿 W 平均
        x_h = x.mean(dim=3, keepdim=True)

        # x_w: (B, C, 1, W)  沿 H 平均
        x_w = x.mean(dim=2, keepdim=True)

        # ---- Step 2: 拼接（在空间维度拼起来）----
        # 为了拼接，把 x_w 转置成 (B, C, W, 1)，这样和 x_h 的最后一维都是 1
        # x_w_t: (B, C, W, 1)
        x_w_t = x_w.permute(0, 1, 3, 2)

        # y: (B, C, H+W, 1)
        y = torch.cat([x_h, x_w_t], dim=2)

        # ---- Step 3: 共享变换（瓶颈）----
        # y: (B, mip, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        # ---- Step 4: 拆分两支 ----
        # y_h: (B, mip, H, 1)
        # y_w: (B, mip, W, 1)
        y_h, y_w = torch.split(y, [h, w], dim=2)

        # 把 y_w 还原回 (B, mip, 1, W)
        y_w = y_w.permute(0, 1, 3, 2)  # (B, mip, 1, W)

        # ---- Step 5: 生成两个方向注意力图 ----
        # a_h: (B, C, H, 1)
        a_h = self.sigmoid(self.conv_h(y_h))

        # a_w: (B, C, 1, W)
        a_w = self.sigmoid(self.conv_w(y_w))

        # ---- Step 6: 重标定（广播逐元素相乘）----
        # a_h 会广播到 W；a_w 会广播到 H
        out = x * a_h * a_w  # (B, C, H, W)

        return out
