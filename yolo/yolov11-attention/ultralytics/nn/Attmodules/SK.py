import torch
import torch.nn as nn
import torch.nn.functional as F


class SK(nn.Module):
    """
    SK (Selective Kernel) / SKConv
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    适配说明：
      - 你的 parse_model 会自动注入 channels=C 作为第一个构造参数，
        所以这里 __init__(channels, ...) 必须以 channels 开头。

    经典 SK 单元流程：Split -> Fuse -> Select
      1) Split: 多分支卷积(不同kernel)提取不同感受野特征
      2) Fuse: 分支特征求和 + GAP 得到全局描述
      3) Select: 用 softmax 在“分支维”上生成权重，对各分支加权融合
    """

    def __init__(
        self,
        channels: int,                 # C（由 parse_model 注入）
        kernels=(3, 5),                # 多分支卷积核大小（常用 3/5 或 3/7）
        reduction: int = 16,           # 压缩比
        L: int = 32,                   # bottleneck 最小维度（论文中常用）
        groups: int = 1,               # 分组卷积（想更轻量可设为 channels 做 depthwise，但需谨慎）
        act: str = "silu",
    ):
        super().__init__()
        assert channels > 0
        self.channels = channels
        self.kernels = list(kernels)
        self.M = len(self.kernels)     # 分支数 M

        # 激活函数
        if act.lower() == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.SiLU(inplace=True)

        # ---- 1) Split: 多分支卷积 ----
        # 每个分支输出 shape 都是 (B, C, H, W)
        self.branches = nn.ModuleList()
        for k in self.kernels:
            p = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k, padding=p, groups=groups, bias=False),
                    nn.BatchNorm2d(channels),
                    self.act,
                )
            )

        # ---- 2) Fuse: 聚合后 GAP ----
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B,C,H,W)->(B,C,1,1)

        # bottleneck 维度 d
        d = max(channels // reduction, L)

        # ---- 3) Select: 生成分支权重 ----
        # (B,C,1,1)->(B,d,1,1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            self.act,
        )
        # 为每个分支生成 (B,C,1,1) 的 logits，之后在分支维 softmax
        self.fc_branches = nn.ModuleList([nn.Conv2d(d, channels, kernel_size=1, bias=True) for _ in range(self.M)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        # ---- Split ----
        # feats: list，每个元素 (B, C, H, W)
        feats = [br(x) for br in self.branches]

        # feats_stack: (B, M, C, H, W)
        feats_stack = torch.stack(feats, dim=1)

        # ---- Fuse ----
        # U: (B, C, H, W)  (各分支求和融合得到全局特征)
        U = feats_stack.sum(dim=1)

        # s: (B, C, 1, 1)
        s = self.gap(U)

        # z: (B, d, 1, 1)
        z = self.fc(s)

        # ---- Select ----
        # logits: (B, M, C, 1, 1)  每个分支一份通道权重logits
        logits = torch.stack([fc(z) for fc in self.fc_branches], dim=1)

        # attn: (B, M, C, 1, 1) 在“分支维 M”上 softmax，得到竞争式权重
        attn = F.softmax(logits, dim=1)

        # ---- Weighted Sum ----
        # out: (B, C, H, W)
        out = (feats_stack * attn).sum(dim=1)
        return out
