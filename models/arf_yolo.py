"""
ARF-YOLO: Attention-Guided Adaptive Resolution-Aware Feature Learning
for UAV Remote Sensing Object Detection.

Main model integration that builds upon YOLOv11m with:
  - AGRH at P4 and P5 levels
  - AMFF replacing bilinear upsampling in the neck
  - FSRA at N3 and N4 output levels
  - ARF-SALoss for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agrh import AGRH
from .amff import AMFF
from .fsra import FSRA
from .arf_saloss import ARFScaleAwareLoss


class ARFYOLO(nn.Module):
    """
    ARF-YOLO model wrapper.

    This module wraps around a YOLOv11m backbone and enhances the neck
    with AGRH, AMFF, and FSRA modules.

    Architecture:
        Backbone: YOLOv11m (Conv, C3k2, SPPF, C2PSA) -> {P3, P4, P5}
        Enhanced Neck:
            1. AGRH at P4 and P5 (before top-down fusion)
            2. AMFF replacing bilinear upsampling (during fusion)
            3. FSRA at N3 and N4 (after fusion, before detection heads)
        Detection Heads: 3-scale anchor-free decoupled heads

    Args:
        num_classes: Number of object categories (default: 10 for VisDrone)
        base_channels: Base channel width (default: 64 for YOLOv11m)
        img_size: Input image size (default: 640)
    """

    def __init__(self, num_classes=10, base_channels=64, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        # Channel widths for YOLOv11m at each FPN level
        # P3: stride 8,  channels = base_channels * 4  = 256
        # P4: stride 16, channels = base_channels * 8  = 512
        # P5: stride 32, channels = base_channels * 16 = 1024
        c3 = base_channels * 4   # 256
        c4 = base_channels * 8   # 512
        c5 = base_channels * 16  # 1024

        # Neck output channels (after fusion)
        neck_c = base_channels * 4  # 256

        # ============================================================
        # AGRH: Applied at P4 and P5 before top-down fusion
        # ============================================================
        # P5-level AGRH: F_L = P5 (c5), F_H = P4 (c4)
        self.agrh_p5 = AGRH(
            in_channels_l=c5,
            in_channels_h=c4,
            out_channels=c5,
            kernel_size=7,
        )
        # P4-level AGRH: F_L = P4 (c4), F_H = P3 (c3)
        self.agrh_p4 = AGRH(
            in_channels_l=c4,
            in_channels_h=c3,
            out_channels=c4,
            kernel_size=7,
        )

        # ============================================================
        # AMFF: Replaces bilinear upsampling in neck fusion
        # ============================================================
        # P5 -> P4 fusion
        self.amff_p5_p4 = AMFF(
            in_channels_deep=c5,
            in_channels_shallow=c4,
            out_channels=c4,
            scale_factor=2,
        )
        # P4 -> P3 fusion
        self.amff_p4_p3 = AMFF(
            in_channels_deep=c4,
            in_channels_shallow=c3,
            out_channels=c3,
            scale_factor=2,
        )

        # ============================================================
        # Post-fusion convolutions (C3k2-style bottleneck blocks)
        # ============================================================
        self.post_fusion_p4 = nn.Sequential(
            nn.Conv2d(c4, neck_c, 1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(neck_c, neck_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
        )
        self.post_fusion_p3 = nn.Sequential(
            nn.Conv2d(c3, neck_c, 1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(neck_c, neck_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
        )

        # Bottom-up path (P3 -> P4 -> P5)
        self.downsample_p3_p4 = nn.Sequential(
            nn.Conv2d(neck_c, neck_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
        )
        self.bu_fuse_p4 = nn.Sequential(
            nn.Conv2d(neck_c * 2, neck_c, 1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(neck_c, neck_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
        )
        self.downsample_p4_p5 = nn.Sequential(
            nn.Conv2d(neck_c, neck_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
        )
        self.bu_fuse_p5 = nn.Sequential(
            nn.Conv2d(neck_c + c5, neck_c, 1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(neck_c, neck_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
        )

        # ============================================================
        # FSRA: Applied at N3 and N4 output levels after fusion
        # ============================================================
        self.fsra = FSRA(
            in_channels=neck_c,
            num_levels=3,
            ca_reduction=4,
            pa_kernel_size=3,
        )

        # ============================================================
        # Detection heads (3-scale, anchor-free, decoupled)
        # ============================================================
        self.detect_heads = nn.ModuleList([
            DetectHead(neck_c, num_classes) for _ in range(3)
        ])

        # ============================================================
        # ARF-SALoss
        # ============================================================
        self.arf_saloss = ARFScaleAwareLoss(alpha=2.0, beta=3.0, lambda_sa=0.5)

    def forward_neck(self, p3, p4, p5):
        """
        Enhanced neck with AGRH, AMFF, and FSRA.

        Args:
            p3: Backbone P3 features (B, C3, H/8, W/8)
            p4: Backbone P4 features (B, C4, H/16, W/16)
            p5: Backbone P5 features (B, C5, H/32, W/32)

        Returns:
            n3, n4, n5: Enhanced neck outputs for detection heads
        """
        # --- AGRH: Dual-resolution enhancement at P4 and P5 ---
        # P5-level: F_L=P5, F_H=P4
        p5_l, p5_h = self.agrh_p5(p5, p4)
        # Use the low-res refined output as the enhanced P5
        # (high-res output contributes to proposal diversity at inference)
        p5_enhanced = p5_l + F.interpolate(
            p5_h, size=p5_l.shape[2:], mode='bilinear', align_corners=False
        )

        # P4-level: F_L=P4, F_H=P3
        p4_l, p4_h = self.agrh_p4(p4, p3)
        p4_enhanced = p4_l + F.interpolate(
            p4_h, size=p4_l.shape[2:], mode='bilinear', align_corners=False
        )

        # --- AMFF: Content-adaptive top-down fusion ---
        # P5 -> P4
        td_p4 = self.amff_p5_p4(p5_enhanced, p4_enhanced)  # (B, C4, H/16, W/16)
        td_p4 = self.post_fusion_p4(td_p4)

        # P4 -> P3
        td_p3 = self.amff_p4_p3(td_p4, p3)  # (B, C3, H/8, W/8)
        n3 = self.post_fusion_p3(td_p3)

        # --- Bottom-up path ---
        # N3 -> N4
        n3_down = self.downsample_p3_p4(n3)
        n4 = self.bu_fuse_p4(torch.cat([n3_down, td_p4], dim=1))

        # N4 -> N5
        n4_down = self.downsample_p4_p5(n4)
        n5 = self.bu_fuse_p5(torch.cat([n4_down, p5_enhanced], dim=1))

        # --- FSRA: Dynamic feature resource allocation ---
        n3, n4, n5 = self.fsra([n3, n4, n5])

        return n3, n4, n5

    def forward(self, features):
        """
        Forward pass through the enhanced neck and detection heads.

        Args:
            features: Tuple of (P3, P4, P5) backbone feature maps.
                      Typically obtained from a YOLOv11 backbone.

        Returns:
            predictions: List of detection outputs from 3 heads
        """
        p3, p4, p5 = features

        # Enhanced neck
        n3, n4, n5 = self.forward_neck(p3, p4, p5)

        # Detection heads
        out3 = self.detect_heads[0](n3)
        out4 = self.detect_heads[1](n4)
        out5 = self.detect_heads[2](n5)

        return [out3, out4, out5]


class DetectHead(nn.Module):
    """
    Decoupled anchor-free detection head.

    Separate classification and regression branches following the
    YOLOv11 decoupled head design.

    Args:
        in_channels: Input channel width
        num_classes: Number of object categories
        reg_max: Maximum regression distribution bins (default: 16)
    """

    def __init__(self, in_channels, num_classes, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        # Classification branch
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)

        # Regression branch
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.reg_pred = nn.Conv2d(in_channels, 4 * reg_max, 1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            dict with 'cls' (B, num_classes, H, W) and 'reg' (B, 4*reg_max, H, W)
        """
        cls_feat = self.cls_branch(x)
        cls_out = self.cls_pred(cls_feat)

        reg_feat = self.reg_branch(x)
        reg_out = self.reg_pred(reg_feat)

        return {"cls": cls_out, "reg": reg_out}
