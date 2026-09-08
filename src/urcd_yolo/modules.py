"""URCD-YOLO Architectural Modules for UAV Pavement Distress Detection.

Implements the four foundational innovations from Yi, Wang & Yan (2026):
1. Improved BiFPN: Softmax-normalized weighted bidirectional feature fusion.
2. WADown: Weighted Adaptive Dual-path Downsampling preserving micro-crack edges.
3. WTConv: Wavelet Transform Convolution capturing low & high frequencies via 2D Haar DWT.
4. MSCA: Multi-Scale Channel Attention targeting linear, elongated distress geometry.
"""

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.conv import Conv


class BiFPN_Concat2(nn.Module):
    """Softmax-normalized weighted fusion for 2 incoming feature maps."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.dimension = dimension
        # Learnable unconstrained weights
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        # This function does Softmax-normalized weighted fusion for 2 feature maps.
        # Input: list of 2 tensors [x0, x1] each (B, C, H, W) -> Intermediate: Softmax weights and spatial size alignment -> Output: concatenated tensor (B, 2*C, H, W).
        if len(x) != 2:
            raise ValueError(f"BiFPN_Concat2 expects exactly 2 inputs, got {len(x)}")

        x0, x1 = x[0], x[1]
        # Align spatial dimensions if off by 1 pixel due to odd spatial sizes
        if x0.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=x0.shape[2:], mode="nearest")

        weights = torch.softmax(self.w, dim=0)
        out = torch.cat([weights[0] * x0, weights[1] * x1], dim=self.dimension)
        return out


class BiFPN_Concat3(nn.Module):
    """Softmax-normalized weighted fusion for 3 incoming feature maps."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.dimension = dimension
        # Learnable unconstrained weights
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        # This function does Softmax-normalized weighted fusion for 3 feature maps.
        # Input: list of 3 tensors [x0, x1, x2] each (B, C, H, W) -> Intermediate: Softmax weights and spatial alignment -> Output: concatenated tensor (B, 3*C, H, W).
        if len(x) != 3:
            raise ValueError(f"BiFPN_Concat3 expects exactly 3 inputs, got {len(x)}")

        x0, x1, x2 = x[0], x[1], x[2]
        target_size = x0.shape[2:]
        if x1.shape[2:] != target_size:
            x1 = F.interpolate(x1, size=target_size, mode="nearest")
        if x2.shape[2:] != target_size:
            x2 = F.interpolate(x2, size=target_size, mode="nearest")

        weights = torch.softmax(self.w, dim=0)
        out = torch.cat(
            [weights[0] * x0, weights[1] * x1, weights[2] * x2],
            dim=self.dimension,
        )
        return out


class WADown(nn.Module):
    """Weighted Adaptive Dual-path Downsampling module.

    Splits input channels into two paths:
    - Path 1: AvgPool followed by 3x3 strided conv (captures smooth context).
    - Path 2: MaxPool followed by 1x1 conv (preserves sharp micro-crack edges).
    Combines both with learnable adaptive Softmax weights.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.c_out_half = c2 // 2
        c_in_half = c1 // 2

        # Path 1: Convolutional feature downsampler
        self.cv1 = Conv(c_in_half, self.c_out_half, k=3, s=2, p=1)
        # Path 2: Edge-preserving max-pool downsampler
        self.cv2 = Conv(c_in_half, self.c_out_half, k=1, s=1, p=0)

        # Learnable adaptive weighting parameter across paths
        self.alpha = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This function does dual-path downsampling with learnable path weighting.
        # Input: tensor x of shape (B, C1, H, W) -> Intermediate: split channels, avg+conv path, max+conv path, Softmax weights -> Output: downsampled tensor (B, C2, H/2, W/2).
        x_pooled = F.avg_pool2d(x, kernel_size=2, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        x1, x2 = x_pooled.chunk(2, dim=1)

        # Apply Path 1
        y1 = self.cv1(x1)

        # Apply Path 2 (MaxPool stride 2, padding 1)
        x2_max = F.max_pool2d(x2, kernel_size=3, stride=2, padding=1)
        y2 = self.cv2(x2_max)

        # Ensure spatial shape alignment if pooling produces a 1-pixel boundary discrepancy
        if y1.shape[2:] != y2.shape[2:]:
            y2 = F.interpolate(y2, size=y1.shape[2:], mode="nearest")

        # Softmax-normalized dynamic balancing (scaled by 2 so neutral expectation equals identity)
        weights = torch.softmax(self.alpha, dim=0) * 2.0
        return torch.cat([weights[0] * y1, weights[1] * y2], dim=1)


class HaarWaveletTransform2D(nn.Module):
    """Orthogonal 2D Discrete Haar Wavelet Transform and Inverse Transform.

    Decomposes spatial features into 4 orthonormal sub-bands:
    - LL: Low-Low approximation (smooth illumination)
    - LH: Low-High horizontal details (vertical cracks)
    - HL: High-Low vertical details (horizontal cracks)
    - HH: High-High diagonal details (fracture junctions)
    """

    def __init__(self):
        super().__init__()
        # 2D Haar basis kernels (fixed, orthonormal, non-trainable)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        # Shape: (4, 1, 2, 2)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def dwt(self, x: torch.Tensor) -> torch.Tensor:
        # This function decomposes spatial features into 4 wavelet frequency sub-bands.
        # Input: tensor x of shape (B, C, H, W) -> Intermediate: depthwise conv with 2D Haar filters -> Output: sub-bands tensor of shape (B, 4*C, H/2, W/2).
        _, c, h, w = x.shape
        pad_h = (2 - (h % 2)) % 2
        pad_w = (2 - (w % 2)) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        filters = self.filters.repeat(c, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        out = F.conv2d(x, filters, stride=2, padding=0, groups=c)
        return out

    def idwt(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        # This function reconstructs spatial features from 4 wavelet frequency sub-bands.
        # Input: sub-bands tensor of shape (B, 4*C, H/2, W/2) -> Intermediate: depthwise transposed conv with Haar filters -> Output: reconstructed tensor (B, C, target_h, target_w).
        _, c4, _, _ = x.shape
        c = c4 // 4
        filters = self.filters.repeat(c, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        rec = F.conv_transpose2d(x, filters, stride=2, padding=0, groups=c)
        return rec[:, :, :target_h, :target_w]


class WTConv2d(nn.Module):
    """Wavelet Transform Convolution Layer.

    Processes features across both the spatial domain and the frequency domain
    via 2D Haar DWT, providing expansive effective receptive fields without
    quadratic parameter growth.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet = HaarWaveletTransform2D()

        # Spatial depthwise convolution
        self.conv_spatial = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )

        # Wavelet sub-band depthwise convolution (operates on 4 sub-bands per channel)
        self.conv_wavelet = nn.Conv2d(
            in_channels * 4,
            in_channels * 4,
            kernel_size=3,
            padding=1,
            groups=in_channels * 4,
            bias=False,
        )

        # Pointwise projection and normalization
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This function applies dual spatial-wavelet convolution for large receptive field crack detection.
        # Input: tensor x of shape (B, Cin, H, W) -> Intermediate: spatial conv + DWT subband conv with IDWT -> Output: projected tensor (B, Cout, H, W).
        _, _, h, w = x.shape

        # 1. Spatial branch
        y_spatial = self.conv_spatial(x)

        # 2. Wavelet branch
        subbands = self.wavelet.dwt(x)
        subbands_proc = self.conv_wavelet(subbands)
        y_wavelet = self.wavelet.idwt(subbands_proc, h, w)

        # 3. Frequency-spatial fusion + projection
        fused = self.act(self.bn(y_spatial + y_wavelet))
        out = self.bn_out(self.proj(fused))
        return out


class Bottleneck_WTConv(nn.Module):
    """Standard bottleneck block with WTConv2d replacing regular 3x3 convolution."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.wtconv = WTConv2d(c_, c2, kernel_size=5)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This function does bottleneck feature transformation with Wavelet Convolution.
        # Input: tensor x (B, C1, H, W) -> Intermediate: 1x1 conv followed by WTConv2d -> Output: residual sum or transformed tensor (B, C2, H, W).
        y = self.wtconv(self.cv1(x))
        return x + y if self.add else y


class C3k2_WTConv(nn.Module):
    """Faster Implementation of C3k2 module embedding WTConv in bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_WTConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This function forwards feature maps through the C3k2-WTConv multi-branch block.
        # Input: tensor x (B, C1, H, W) -> Intermediate: 1x1 conv, chunking, sequential Bottleneck_WTConv -> Output: concatenated 1x1 projected tensor (B, C2, H, W).
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MSCA(nn.Module):
    """Multi-Scale Channel Attention module.

    Applies cross-domain multi-scale strip convolutions (1x7, 7x1, 1x11, 11x1)
    to trace thin linear cracks across varying spatial orientations,
    followed by Squeeze-and-Excitation channel gating.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        hidden = max(8, channels // reduction)

        # 1. Local spatial aggregation
        self.conv_local = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)

        # 2. Multi-scale strip convolutions (horizontal & vertical crack alignment)
        self.strip7_1 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False)
        self.strip7_2 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels, bias=False)

        self.strip11_1 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels, bias=False)
        self.strip11_2 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels, bias=False)

        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # 3. Channel Attention gating MLP
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This function computes multi-scale channel attention and recalibrates input features.
        # Input: tensor x of shape (B, C, H, W) -> Intermediate: local conv, multi-scale strip convolutions, GAP, MLP channel gating -> Output: recalibrated tensor (B, C, H, W).
        b, c, _, _ = x.shape

        # Step 1: Local feature extraction
        u = self.conv_local(x)

        # Step 2: Multi-scale strip convolutions
        scale1 = self.strip7_2(self.strip7_1(u))
        scale2 = self.strip11_2(self.strip11_1(u))
        scale3 = self.conv1x1(u)

        fused = scale1 + scale2 + scale3 + u

        # Step 3: Squeeze-and-Excitation Channel Attention
        gap = self.gap(fused).view(b, c)
        attn_channel = self.fc(gap).view(b, c, 1, 1)

        # Step 4: Cross-domain modulation
        return x * attn_channel
