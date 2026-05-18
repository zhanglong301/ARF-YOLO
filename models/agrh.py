"""
Attention-Guided Resolution Heads (AGRH) with Multi-Perspective Feature Attention (MPFA).

AGRH is applied at P4 and P5 FPN levels before the top-down fusion path.
For each level, it processes dual-resolution streams:
  - F_L: low-resolution semantic stream (current FPN level)
  - F_H: high-resolution spatial stream (adjacent higher-resolution FPN level)
Each stream is refined by MPFA and forwarded to a dedicated detection sub-head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPFA(nn.Module):
    """
    Multi-Perspective Feature Attention.

    Computes spatial attention along three perspectives:
      Perspective 1: Original H x W
      Perspective 2: Height-transposed (C x W view)
      Perspective 3: Width-transposed (C x H view)

    Each perspective uses dual-pooling (AvgPool + MaxPool) followed by
    a 7x7 depthwise convolution. The three attention maps are summed
    and sigmoid-gated for multiplicative feature recalibration.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2

        # Perspective 1: standard spatial attention (H x W)
        self.conv_p1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        # Perspective 2: height-transposed attention
        self.conv_p2 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        # Perspective 3: width-transposed attention
        self.conv_p3 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def _pool_and_conv(self, x, conv_layer):
        """Apply dual-pooling and convolution to compute attention map."""
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        pooled = torch.cat([avg_out, max_out], dim=1)    # (B, 2, H, W)
        return conv_layer(pooled)                        # (B, 1, H, W)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Attention-refined feature: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Perspective 1: Original H x W
        A1 = self._pool_and_conv(x, self.conv_p1)  # (B, 1, H, W)

        # Perspective 2: Height-transposed -> (B, H, C, W) -> pool on dim=1(H)
        x_ht = x.permute(0, 2, 1, 3)  # (B, H, C, W)
        A2 = self._pool_and_conv(x_ht, self.conv_p2)  # (B, 1, C, W)
        A2 = A2.permute(0, 2, 1, 3)  # (B, C, 1, W) -> broadcast to (B, 1, H, W)
        # Expand A2 to match spatial dims
        A2 = A2.expand_as(A1) if A2.shape == A1.shape else \
            F.interpolate(A2, size=(H, W), mode='bilinear', align_corners=False)

        # Perspective 3: Width-transposed -> (B, W, H, C) -> pool on dim=1(W)
        x_wt = x.permute(0, 3, 2, 1)  # (B, W, H, C)
        A3 = self._pool_and_conv(x_wt, self.conv_p3)  # (B, 1, H, C)
        A3 = A3.permute(0, 3, 2, 1)  # (B, C, H, 1) -> broadcast to (B, 1, H, W)
        A3 = A3.expand_as(A1) if A3.shape == A1.shape else \
            F.interpolate(A3, size=(H, W), mode='bilinear', align_corners=False)

        # Aggregate and apply
        attention = self.sigmoid(A1 + A2 + A3)  # (B, 1, H, W)
        return x * attention


class AGRH(nn.Module):
    """
    Attention-Guided Resolution Heads.

    Processes dual-resolution streams (F_L and F_H) through parallel MPFA
    modules. F_L is the backbone feature at the current FPN level; F_H is
    sourced from the adjacent higher-resolution FPN level.

    A 1x1 convolution aligns F_H to the same channel dimension as F_L
    when their channel widths differ.

    Args:
        in_channels_l: Channel width of F_L (low-resolution semantic stream)
        in_channels_h: Channel width of F_H (high-resolution spatial stream)
        out_channels: Output channel width (same for both branches)
        kernel_size: Kernel size for MPFA depthwise convolution (default: 7)
    """

    def __init__(self, in_channels_l, in_channels_h, out_channels, kernel_size=7):
        super().__init__()

        # Channel alignment for F_H if needed
        self.align_h = nn.Identity()
        if in_channels_h != out_channels:
            self.align_h = nn.Conv2d(in_channels_h, out_channels, 1, bias=False)

        # Channel alignment for F_L if needed
        self.align_l = nn.Identity()
        if in_channels_l != out_channels:
            self.align_l = nn.Conv2d(in_channels_l, out_channels, 1, bias=False)

        # MPFA for each stream
        self.mpfa_l = MPFA(kernel_size=kernel_size)
        self.mpfa_h = MPFA(kernel_size=kernel_size)

        # Detection sub-heads (lightweight conv heads)
        self.head_l = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.head_h = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, f_l, f_h):
        """
        Args:
            f_l: Low-resolution semantic feature (B, C_l, H_l, W_l)
            f_h: High-resolution spatial feature (B, C_h, H_h, W_h), H_h = 2*H_l

        Returns:
            out_l: MPFA-refined low-res features (B, C_out, H_l, W_l)
            out_h: MPFA-refined high-res features (B, C_out, H_h, W_h)
        """
        # Channel alignment
        f_l = self.align_l(f_l)
        f_h = self.align_h(f_h)

        # MPFA attention refinement
        f_l = self.mpfa_l(f_l)
        f_h = self.mpfa_h(f_h)

        # Detection sub-heads
        out_l = self.head_l(f_l)
        out_h = self.head_h(f_h)

        return out_l, out_h
