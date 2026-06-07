import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_gn_groups(C: int, prefer: int = 4) -> int:
    """
    论文里写 GroupNorm-4，这里做个工程兼容：
    - 若 C % 4 == 0，用 4
    - 否则用一个能整除 C 的最大组数（<=prefer），再不行退化为 1
    """
    if C % prefer == 0:
        return prefer
    for g in range(prefer, 0, -1):
        if C % g == 0:
            return g
    return 1


class MS_DWConv1d(nn.Module):
    """
    论文图右侧的 MS-DWConv1d(k1,k2,k3,k4)：
    - 输入:  (B, C, L)
    - 输出:  (B, C, L)
    - depthwise 1D conv（groups=C）
    """
    def __init__(self, channels: int, k: int):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(x)


class CA_SHSA(nn.Module):
    """
    论文图 PCSA 里的 CA-SHSA（Channel-wise Single-Head Self-Attention）
    这里按图实现一个“通道维单头自注意力”，注意力在通道维 C 上计算。

    输入:  (B, C, Hp, Wp)
    过程:
      - 1x1 DWConv2d 生成 Q/K/V（图中 3 个 DWConv2d 1x1）
      - reshape 到 (B, C, N) 其中 N=Hp*Wp
      - Attn = softmax( Q @ K^T )  -> (B, C, C)
      - Out  = Attn @ V            -> (B, C, N) -> reshape 回 (B,C,Hp,Wp)

    输出:  (B, C, Hp, Wp)
    """
    def __init__(self, channels: int):
        super().__init__()
        # DWConv2d 1x1（groups=C）：每个通道独立映射（贴合图）
        self.q = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Hp, Wp = x.shape
        N = Hp * Wp

        q = self.q(x).view(B, C, N)  # (B,C,N)
        k = self.k(x).view(B, C, N)  # (B,C,N)
        v = self.v(x).view(B, C, N)  # (B,C,N)

        # 通道注意力： (B,C,N) @ (B,N,C) -> (B,C,C)
        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(max(1, N))
        attn = F.softmax(attn, dim=-1)

        # (B,C,C) @ (B,C,N) -> (B,C,N)
        out = torch.bmm(attn, v).view(B, C, Hp, Wp)
        return out


class SCSA(nn.Module):
    """
    SCSA: Spatial and Channel Synergistic Attention
    年份：2024（论文/预印本提出）

    输入/输出: (B, C, H, W) -> (B, C, H, W)

    按论文图实现：
      Part-1: SMSA（Shared Multi-Semantic Spatial Attention）
        - X AvgPool: 沿 W 平均 -> (B,C,H)
        - Y AvgPool: 沿 H 平均 -> (B,C,W)
        - split（n=4）：按通道分组
        - 每组用不同核大小 MS-DWConv1d（3,5,7,9）
        - Concat 拼回 (B,C,H)/(B,C,W)
        - GroupNorm-4 -> Sigmoid 得到 Ah(B,C,H), Aw(B,C,W)
        - 广播融合：Ms = Ah.unsqueeze(-1) * Aw.unsqueeze(-2) -> (B,C,H,W)
        - X_s = X * Ms

      Part-2: PCSA（Progressive Channel-wise Self-Attention）
        - AvgPool: 压缩到 (Hp,Wp)（论文图中的 H'×W'）
        - GroupNorm-1
        - CA-SHSA（通道单头自注意力）
        - AvgPool -> Sigmoid 得到 Mc(B,C,1,1)
        - Out = X_s * Mc
    """

    def __init__(
        self,
        channels: int,                 # 必须第一个参数：匹配你的 parse_model
        n: int = 4,                    # 论文图 n=4
        kernels=(3, 5, 7, 9),          # 论文图 (3,5,7,9)
        pool_hw: int = 7,              # PCSA 的 H'、W'（越小越省算力）
    ):
        super().__init__()
        assert channels > 0
        assert n > 0
        assert channels % n == 0, f"channels={channels} 必须能被 n={n} 整除（用于 split）"

        self.C = channels
        self.n = n
        self.Cg = channels // n
        if isinstance(kernels, (list, tuple)):
            assert len(kernels) == n, "kernels 长度需等于 n（每组一个核）"
        else:
            raise TypeError("kernels 必须是 list/tuple")

        # ---- SMSA：两个方向各一套 MS-DWConv1d（与图一致）----
        self.ms_h = nn.ModuleList([MS_DWConv1d(self.Cg, int(k)) for k in kernels])  # 处理 (B,Cg,H)
        self.ms_w = nn.ModuleList([MS_DWConv1d(self.Cg, int(k)) for k in kernels])  # 处理 (B,Cg,W)

        # 图里是 GroupNorm-4（作用在 B×C×H / B×C×W）
        g = _safe_gn_groups(channels, prefer=4)
        self.gn_h = nn.GroupNorm(g, channels)
        self.gn_w = nn.GroupNorm(g, channels)
        self.sigmoid = nn.Sigmoid()

        # ---- PCSA：AvgPool 到 H'×W' + GN-1 + CA-SHSA ----
        self.pool_hw = int(pool_hw)
        self.gn1 = nn.GroupNorm(1, channels)   # GroupNorm-1（图中写的）
        self.ca_shsa = CA_SHSA(channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 输出 Mc: (B,C,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.C

        # =========================================================
        # Part-1: SMSA（按图：先两方向池化，再 split，再 MS-DWConv1d，再 concat，再 GN+Sigmoid）
        # =========================================================
        # X AvgPool（沿 W 平均）: (B,C,H,W) -> (B,C,H)
        x_h = x.mean(dim=3)  # (B, C, H)

        # Y AvgPool（沿 H 平均）: (B,C,H,W) -> (B,C,W)
        x_w = x.mean(dim=2)  # (B, C, W)

        # split：按通道切成 n 组
        x_h_chunks = torch.chunk(x_h, self.n, dim=1)  # n 个 (B,Cg,H)
        x_w_chunks = torch.chunk(x_w, self.n, dim=1)  # n 个 (B,Cg,W)

        # 每组做不同 kernel 的 MS-DWConv1d
        h_out = [self.ms_h[i](x_h_chunks[i]) for i in range(self.n)]  # n 个 (B,Cg,H)
        w_out = [self.ms_w[i](x_w_chunks[i]) for i in range(self.n)]  # n 个 (B,Cg,W)

        # Concat：拼回 B×C×H / B×C×W
        h_cat = torch.cat(h_out, dim=1)  # (B, C, H)
        w_cat = torch.cat(w_out, dim=1)  # (B, C, W)

        # GroupNorm-4 + Sigmoid：得到 Ah / Aw
        Ah = self.sigmoid(self.gn_h(h_cat))  # (B, C, H)
        Aw = self.sigmoid(self.gn_w(w_cat))  # (B, C, W)

        # 广播融合：Ms = Ah ⊗ Aw -> (B,C,H,W)
        Ms = Ah.unsqueeze(-1) * Aw.unsqueeze(-2)  # (B, C, H, W)

        # 作用到输入
        x_s = x * Ms  # (B, C, H, W)

        # =========================================================
        # Part-2: PCSA（按图：AvgPool->GN-1->CA-SHSA->AvgPool->Sigmoid->乘回去）
        # =========================================================
        # AvgPool 到 H'×W'（图中 H'×W'）
        p = self.pool_hw
        xp = F.adaptive_avg_pool2d(x_s, (p, p))  # (B, C, p, p)

        # GroupNorm-1
        xp = self.gn1(xp)  # (B, C, p, p)

        # CA-SHSA（通道单头自注意力）
        xp = self.ca_shsa(xp)  # (B, C, p, p)

        # AvgPool -> Sigmoid 得到通道权重 Mc
        Mc = torch.sigmoid(self.avgpool(xp))  # (B, C, 1, 1)

        # 输出
        out = x_s * Mc  # (B, C, H, W)
        return out
