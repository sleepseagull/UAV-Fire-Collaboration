import torch
import torch.nn as nn


class SLAM(nn.Module):
    """
    SLAM: Spatial Location Attention Module (2025)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    三个分支生成注意力图：
      1) 垂直方向: M_H 形状 (B, C, 1, W)   —— pool over H
      2) 水平方向: M_W 形状 (B, C, H, 1)   —— pool over W
      3) 空间方向: M_S 形状 (B, 1, H, W)   —— pool over C

    最终输出:
      out = x * M_H * M_W * M_S    (利用广播机制 broadcast)
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: 输入通道数 C（⚠️必须放第一个，匹配 parse_model 的注入）
            reduction: 1x1瓶颈压缩比（W0/W1）
            kernel_size: 空间分支卷积核大小，论文常用 7
        """
        super().__init__()
        assert channels > 0
        hidden = max(1, channels // reduction)

        # --------- H/W 方向分支的共享结构：W0 -> ReLU -> W1（论文写作 W0/W1）---------
        # 用 1x1 Conv2d 实现：对 (B,C,1,W) 或 (B,C,H,1) 都能工作
        self.hw_conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)  # W0: C -> hidden
        self.hw_act = nn.ReLU(inplace=True)
        self.hw_conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)  # W1: hidden -> C

        # --------- S（空间）分支：7x7 卷积（论文写作 f_{7x7}^{DW}）---------
        pad = kernel_size // 2
        self.s_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=pad, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape

        # ============================================================
        # 1) 垂直方向注意力 M_H : (B, C, 1, W)
        #    = sigmoid( W1(W0( F_H_avg + F_H_max )) )
        #    这里“pool over H”，保留 W
        # ============================================================
        fh_avg = x.mean(dim=2, keepdim=True)                 # (B, C, 1, W)
        fh_max = torch.max(x, dim=2, keepdim=True)[0]        # (B, C, 1, W)
        fh = fh_avg + fh_max                                 # ⊕  (B, C, 1, W)

        mh = self.hw_conv2(self.hw_act(self.hw_conv1(fh)))   # (B, C, 1, W)
        mh = self.sigmoid(mh)                                 # σ  (B, C, 1, W)

        # ============================================================
        # 2) 水平方向注意力 M_W : (B, C, H, 1)
        #    = sigmoid( W1(W0( F_W_avg + F_W_max )) )
        #    这里“pool over W”，保留 H
        # ============================================================
        fw_avg = x.mean(dim=3, keepdim=True)                 # (B, C, H, 1)
        fw_max = torch.max(x, dim=3, keepdim=True)[0]        # (B, C, H, 1)
        fw = fw_avg + fw_max                                 # ⊕  (B, C, H, 1)

        mw = self.hw_conv2(self.hw_act(self.hw_conv1(fw)))   # (B, C, H, 1)
        mw = self.sigmoid(mw)                                 # σ  (B, C, H, 1)

        # ============================================================
        # 3) 空间注意力 M_S : (B, 1, H, W)
        #    = sigmoid( f_{7x7}( F_S_avg + F_S_max ) )
        #    这里“pool over C”，得到 1×H×W
        # ============================================================
        fs_avg = x.mean(dim=1, keepdim=True)                 # (B, 1, H, W)
        fs_max = torch.max(x, dim=1, keepdim=True)[0]        # (B, 1, H, W)
        fs = fs_avg + fs_max                                 # ⊕  (B, 1, H, W)

        ms = self.s_conv(fs)                                  # (B, 1, H, W)
        ms = self.sigmoid(ms)                                 # σ  (B, 1, H, W)

        # ============================================================
        # 4) 逐元素相乘（⊗），依赖广播：
        #    mh 会在 H 维广播，mw 会在 W 维广播，ms 会在 C 维广播
        # ============================================================
        out = x * mh * mw * ms                                # (B, C, H, W)
        return out
