"""
Adaptive Multi-Level Feature Fusion Module (AMFF).

AMFF replaces the two bilinear upsampling operations in the standard YOLOv11
neck with content-adaptive upsampling, preserving spatial detail during
cross-level feature merging. It consists of three sub-components:

  - FAUS (Feature-Adaptive Upsampling): Content-conditioned dynamic kernel
    generation for upsampling deep features.
  - FRS (Feature Refinement from Shallow): Structure-guided refinement of
    shallow features using edge and texture cues.
  - AFFS (Adaptive Feature Fusion with Selection): Learned channel-wise
    weighting for cross-level feature combination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FAUS(nn.Module):
    """
    Feature-Adaptive Upsampling.

    Generates content-conditioned dynamic upsampling kernels from the input
    feature map, then applies them via depth-wise deformable convolution
    to produce spatially upsampled features that preserve content structure.

    Args:
        in_channels: Number of input channels
        scale_factor: Upsampling factor (default: 2)
        kernel_size: Size of the dynamic kernel (default: 3)
    """

    def __init__(self, in_channels, scale_factor=2, kernel_size=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Kernel generation branch: predict per-pixel upsampling kernels
        k2 = kernel_size * kernel_size
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, k2 * scale_factor * scale_factor, 1, bias=False),
        )

        # Learnable upsampling weight (pixel-shuffle style)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        # Post-refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Upsampled feature: (B, C, H*scale, W*scale)
        """
        B, C, H, W = x.shape

        # Generate dynamic kernels
        kernels = self.kernel_gen(x)  # (B, k2*s2, H, W)

        # Standard bilinear upsampling as base
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear',
                             align_corners=False)  # (B, C, H*s, W*s)

        # Unfold neighborhoods from upsampled feature
        x_unfold = F.unfold(x_up, self.kernel_size, padding=self.padding)
        # (B, C*k2, H_up*W_up)

        H_up, W_up = H * self.scale_factor, W * self.scale_factor

        # Reshape kernels via pixel shuffle -> (B, k2, H_up, W_up)
        k2 = self.kernel_size * self.kernel_size
        kernels = kernels.view(B, k2, self.scale_factor, self.scale_factor, H, W)
        kernels = kernels.permute(0, 1, 4, 2, 5, 3).contiguous()
        kernels = kernels.view(B, k2, H_up, W_up)
        kernels = F.softmax(kernels, dim=1)  # Normalize kernels

        # Reshape unfolded features
        x_unfold = x_unfold.view(B, C, k2, H_up * W_up)

        # Apply dynamic kernels
        kernels_flat = kernels.view(B, 1, k2, H_up * W_up)
        out = (x_unfold * kernels_flat).sum(dim=2)  # (B, C, H_up*W_up)
        out = out.view(B, C, H_up, W_up)

        # Post-refinement
        out = self.refine(out)

        return out


class FRS(nn.Module):
    """
    Feature Refinement from Shallow.

    Refines shallow (high-resolution) features using edge and texture cues
    through a lightweight convolutional path, preparing them for fusion with
    upsampled deep features.

    Args:
        in_channels: Number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: Shallow feature (B, C, H, W)
        Returns:
            Refined feature: (B, C, H, W)
        """
        return x + self.refine(x)


class AFFS(nn.Module):
    """
    Adaptive Feature Fusion with Selection.

    Computes channel-wise attention weights for combining deep (upsampled)
    and shallow (refined) features. Uses a squeeze-excitation style mechanism
    to learn the optimal per-channel weighting.

    Args:
        in_channels: Number of channels per input branch
        reduction: Channel reduction ratio for SE block (default: 4)
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels * 2, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_deep, f_shallow):
        """
        Args:
            f_deep: Upsampled deep feature (B, C, H, W)
            f_shallow: Refined shallow feature (B, C, H, W)
        Returns:
            Fused feature: (B, C, H, W)
        """
        B, C, H, W = f_deep.shape

        # Concatenate and squeeze
        combined = torch.cat([f_deep, f_shallow], dim=1)  # (B, 2C, H, W)
        weights = self.gap(combined)  # (B, 2C, 1, 1)
        weights = self.fc(weights)    # (B, 2C, 1, 1)
        weights = self.sigmoid(weights)

        # Split weights for deep and shallow
        w_deep, w_shallow = weights.split(C, dim=1)

        # Weighted fusion
        out = f_deep * w_deep + f_shallow * w_shallow

        return out


class AMFF(nn.Module):
    """
    Adaptive Multi-Level Feature Fusion Module.

    Combines FAUS, FRS, and AFFS to perform content-adaptive cross-level
    feature fusion, replacing standard bilinear upsampling in the FPN neck.

    Args:
        in_channels_deep: Channels of the deep (low-res) feature
        in_channels_shallow: Channels of the shallow (high-res) feature
        out_channels: Output channels after fusion
        scale_factor: Upsampling factor for FAUS (default: 2)
    """

    def __init__(self, in_channels_deep, in_channels_shallow, out_channels,
                 scale_factor=2):
        super().__init__()

        # Align channels if needed
        self.align_deep = nn.Identity()
        if in_channels_deep != out_channels:
            self.align_deep = nn.Sequential(
                nn.Conv2d(in_channels_deep, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.align_shallow = nn.Identity()
        if in_channels_shallow != out_channels:
            self.align_shallow = nn.Sequential(
                nn.Conv2d(in_channels_shallow, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # Sub-components
        self.faus = FAUS(out_channels, scale_factor=scale_factor)
        self.frs = FRS(out_channels)
        self.affs = AFFS(out_channels)

    def forward(self, f_deep, f_shallow):
        """
        Args:
            f_deep: Deep semantic feature (B, C_d, H, W)
            f_shallow: Shallow spatial feature (B, C_s, 2H, 2W)
        Returns:
            Fused feature: (B, C_out, 2H, 2W)
        """
        # Channel alignment
        f_deep = self.align_deep(f_deep)
        f_shallow = self.align_shallow(f_shallow)

        # Content-adaptive upsampling
        f_up = self.faus(f_deep)  # (B, C_out, 2H, 2W)

        # Shallow feature refinement
        f_ref = self.frs(f_shallow)  # (B, C_out, 2H, 2W)

        # Adaptive fusion
        out = self.affs(f_up, f_ref)  # (B, C_out, 2H, 2W)

        return out
