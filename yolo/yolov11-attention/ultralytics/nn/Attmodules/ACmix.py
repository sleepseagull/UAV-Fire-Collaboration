import torch
import torch.nn as nn
import torch.nn.functional as F


class ACmix(nn.Module):
    """
    ACmix (CVPR 2022): shared 1x1 projections + (local) self-attn branch + shift/conv branch, then fuse by alpha/beta.
    Input/Output: (B, C, H, W) -> (B, C, H, W)

    工程说明（适合 YOLO 缝合）：
    - 为了避免全局注意力 O((HW)^2) 爆显存，这里实现的是 “局部窗口(k×k)注意力”，复杂度 ~ O(C*k^2*H*W)。
    - Shift/Conv 分支用 unfold 等价实现 “shift+sum”（图里的 Shift Operation）。
    """

    def __init__(self, channels: int, kernel_size: int = 3, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels 必须能被 num_heads 整除"
        self.C = channels
        self.h = num_heads
        self.d = channels // num_heads  # 每个 head 的通道数
        self.k = kernel_size
        self.k2 = kernel_size * kernel_size
        self.pad = kernel_size // 2

        # ---- ① 共享投影：Conv1x1 产生 Q,K,V ----
        # qkv: (B, 3C, H, W) -> split -> q/k/v: (B, C, H, W)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=False)

        # ---- ② Conv(shift) 分支：图中 Fully Connected（论文/代码里常用 group 1x1 conv 实现）----
        # 输入 concat(q,k,v): (B, 3C, H, W)
        # 输出 kernel: (B, h*k^2, H, W)  -> reshape -> (B, h, k^2, H*W)
        # 说明：论文里 FC 形式有过勘误，代码里是以 group conv 做到按 head 分组生成权重。:contentReference[oaicite:1]{index=1}
        self.kernel_gen = nn.Conv2d(3 * channels, num_heads * self.k2, kernel_size=1, groups=num_heads, bias=True)

        # ---- ③ 融合系数 α、β（图右侧的 xα 和 xβ）----
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # 可选：输出再做一次 1x1 投影（工程里常见）
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        L = H * W

        # ========== ① 共享 1x1 投影 ==========
        qkv = self.qkv(x)                 # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)     # each: (B, C, H, W)

        # 统一转成 multi-head 视角（便于后续计算）
        # q_flat: (B, h, d, L)
        q_flat = q.view(B, self.h, self.d, L)

        # 把 k、v 展开成 k×k 邻域 patch（unfold 等价图里的 shift/gather）
        # unfold: (B, C*k^2, L) -> (B, h, d, k^2, L)
        k_patch = F.unfold(k, kernel_size=self.k, padding=self.pad).view(B, self.h, self.d, self.k2, L)
        v_patch = F.unfold(v, kernel_size=self.k, padding=self.pad).view(B, self.h, self.d, self.k2, L)

        # ========== ② Attention 分支（图下半部分：Q/K/V -> Attention Operation -> Concatenation） ==========
        # logits: (B, h, k^2, L)  （对每个位置，只在 k^2 邻域里 softmax）
        logits = (q_flat.unsqueeze(3) * k_patch).sum(dim=2)     # sum over d
        attn = F.softmax(logits, dim=2)                         # (B, h, k^2, L)

        # attn_out: (B, h, d, L) -> (B, C, H, W)
        attn_out = (attn.unsqueeze(2) * v_patch).sum(dim=3)     # sum over k^2
        attn_out = attn_out.view(B, C, H, W)

        # ========== ③ Conv(shift) 分支（图上半部分：FC -> Shift Operation -> Concatenation） ==========
        # kernel: (B, h*k^2, H, W) -> (B, h, k^2, L)
        kernel = self.kernel_gen(torch.cat([q, k, v], dim=1)).view(B, self.h, self.k2, L)

        # conv_out: (B, h, d, L) -> (B, C, H, W)
        # 这里 kernel 在 d 维上广播（同一 head 的各通道共享同一组 k^2 权重），符合“shift+sum”的直觉实现
        conv_out = (kernel.unsqueeze(2) * v_patch).sum(dim=3)   # sum over k^2
        conv_out = conv_out.view(B, C, H, W)

        # ========== ④ 融合（图右侧：xα、xβ 再相加） ==========
        out = self.alpha * conv_out + self.beta * attn_out      # (B, C, H, W)
        out = self.proj(out)                                    # (B, C, H, W)
        return out
