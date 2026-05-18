"""
Fast Scale Resource Assigner (FSRA).

Adapted from infrared small target detection to multi-category UAV detection.
Dynamically reallocates feature representation capacity along three orthogonal
dimensions: channels, spatial patches, and scale levels.

Key adaptations from the original infrared version:
  1. Channel Assigner: compression ratio reduced from r=8 to r=4 for
     category-diverse channel importance estimation.
  2. Patch Assigner: GCA kernel size increased from k=1 to k=3 for
     broader spatial context in scale-diverse imagery.
  3. Frame Assigner: extended from L=2 to L=3 scale levels with softmax
     normalization for competitive scale-level allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAssigner(nn.Module):
    """
    Channel Assigner (CA).

    Estimates per-channel importance via squeeze-excitation with reduced
    compression ratio (r=4) to handle 10+ UAV object categories.

    s_CA = sigmoid(FC2(ReLU(FC1(GAP(B))))) in R^{LC}

    Args:
        total_channels: Total number of channels (L * C)
        reduction: Compression ratio (default: 4)
    """

    def __init__(self, total_channels, reduction=4):
        super().__init__()
        mid_channels = max(total_channels // reduction, 16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(total_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, total_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Concatenated multi-scale features (B, L*C, H, W)
        Returns:
            Channel importance weights (B, L*C, 1, 1)
        """
        B, LC, _, _ = x.shape
        y = self.gap(x).view(B, LC)
        y = self.fc(y).view(B, LC, 1, 1)
        return y


class PatchAssigner(nn.Module):
    """
    Patch Assigner (PA).

    Estimates spatial importance via global context aggregation (GCA)
    with k=3 kernel for broader receptive field in UAV imagery.

    s_PA = sigmoid(Conv_1x1(GCA(B_spatial))) in R^{1 x h x w}

    Args:
        in_channels: Number of input channels
        kernel_size: GCA kernel size (default: 3)
    """

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.gca = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Features (B, C, H, W)
        Returns:
            Spatial importance weights (B, 1, H, W)
        """
        y = self.gca(x)
        y = self.conv(y)
        return y


class FrameAssigner(nn.Module):
    """
    Frame Assigner (FA).

    Estimates per-scale-level importance. Extended from L=2 (infrared)
    to L=3 (YOLO FPN) with softmax normalization for competitive allocation.

    s_FA = softmax(FC(Avg_{h,w}(B))) in R^L

    Args:
        in_channels: Number of channels per scale level
        num_levels: Number of FPN scale levels (default: 3)
    """

    def __init__(self, in_channels, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * num_levels, num_levels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features_list):
        """
        Args:
            features_list: List of L features, each (B, C, H_l, W_l)
        Returns:
            Scale-level importance weights: list of L scalars, each (B, 1, 1, 1)
        """
        B = features_list[0].shape[0]

        # Global average pooling for each level
        pooled = []
        for f in features_list:
            pooled.append(self.avg(f).view(B, -1))  # (B, C)
        pooled = torch.cat(pooled, dim=1)  # (B, L*C)

        # Compute per-level weights
        weights = self.fc(pooled)  # (B, L)

        # Split into per-level weights
        weight_list = []
        for i in range(self.num_levels):
            w = weights[:, i].view(B, 1, 1, 1)
            weight_list.append(w)

        return weight_list


class FSRA(nn.Module):
    """
    Fast Scale Resource Assigner.

    Combines Channel Assigner, Patch Assigner, and Frame Assigner to
    dynamically reallocate representational capacity along three orthogonal
    dimensions. Uses additive skip connections to preserve original features.

    B_hat^l = B^l + (s_CA^l * B^l) + (s_PA * B^l) + (s_FA^l * B^l)

    Args:
        in_channels: Number of channels per scale level
        num_levels: Number of FPN scale levels (default: 3)
        ca_reduction: Channel Assigner compression ratio (default: 4)
        pa_kernel_size: Patch Assigner GCA kernel size (default: 3)
    """

    def __init__(self, in_channels, num_levels=3, ca_reduction=4, pa_kernel_size=3):
        super().__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels

        # Three parallel assigners
        self.channel_assigner = ChannelAssigner(
            total_channels=in_channels * num_levels,
            reduction=ca_reduction,
        )
        self.patch_assigner = PatchAssigner(
            in_channels=in_channels,
            kernel_size=pa_kernel_size,
        )
        self.frame_assigner = FrameAssigner(
            in_channels=in_channels,
            num_levels=num_levels,
        )

    def forward(self, features_list):
        """
        Args:
            features_list: List of L features, each (B, C, H_l, W_l).
                           All features must be resized to the same spatial
                           resolution before calling this method.

        Returns:
            List of L reallocated features, each (B, C, H_l, W_l)
        """
        B = features_list[0].shape[0]
        C = self.in_channels

        # Resize all features to the same spatial resolution (largest level)
        target_h = max(f.shape[2] for f in features_list)
        target_w = max(f.shape[3] for f in features_list)

        aligned = []
        for f in features_list:
            if f.shape[2] != target_h or f.shape[3] != target_w:
                f_aligned = F.interpolate(f, size=(target_h, target_w),
                                          mode='bilinear', align_corners=False)
            else:
                f_aligned = f
            aligned.append(f_aligned)

        # Concatenate along channel dimension for Channel Assigner
        concat = torch.cat(aligned, dim=1)  # (B, L*C, H, W)

        # Channel Assigner: per-channel importance
        s_ca = self.channel_assigner(concat)  # (B, L*C, 1, 1)

        # Patch Assigner: spatial importance (computed on mean across levels)
        mean_feat = torch.mean(concat, dim=1, keepdim=True)
        mean_feat = mean_feat.expand(-1, C, -1, -1)
        s_pa = self.patch_assigner(mean_feat)  # (B, 1, H, W)

        # Frame Assigner: per-level importance
        s_fa_list = self.frame_assigner(aligned)  # List of (B, 1, 1, 1)

        # Apply assignments with additive skip connection
        outputs = []
        for l in range(self.num_levels):
            b_l = aligned[l]  # (B, C, H, W)

            # Channel assignment slice for this level
            s_ca_l = s_ca[:, l * C:(l + 1) * C, :, :]  # (B, C, 1, 1)

            # Additive modulation
            out_l = b_l + (s_ca_l * b_l) + (s_pa * b_l) + (s_fa_list[l] * b_l)

            # Resize back to original spatial resolution if needed
            orig_h, orig_w = features_list[l].shape[2], features_list[l].shape[3]
            if out_l.shape[2] != orig_h or out_l.shape[3] != orig_w:
                out_l = F.interpolate(out_l, size=(orig_h, orig_w),
                                      mode='bilinear', align_corners=False)
            outputs.append(out_l)

        return outputs
