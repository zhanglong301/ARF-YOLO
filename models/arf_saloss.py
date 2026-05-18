"""
ARF-Scale-Aware Loss (ARF-SALoss).

Augments the standard YOLO training objective with an instance-level
inverse-scale weighting term that amplifies the gradient contribution
of small instances by up to 3x.

L_sa = (1/N) * sum_i w(a_i) * L_CIoU(b_hat_i, b_i*)
w(a_i) = 1 + alpha * exp(-beta * a_i / a_bar)

Default hyperparameters: alpha=2.0, beta=3.0, lambda=0.5
"""

import torch
import torch.nn as nn
import math


def bbox_area(boxes):
    """
    Compute area of bounding boxes.

    Args:
        boxes: (N, 4) in format [x1, y1, x2, y2] or [cx, cy, w, h]

    Returns:
        areas: (N,)
    """
    if boxes.shape[-1] == 4:
        # Assume xywh format
        return boxes[..., 2] * boxes[..., 3]
    return boxes


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU Loss.

    Args:
        pred_boxes: (N, 4) in [x1, y1, x2, y2] format
        target_boxes: (N, 4) in [x1, y1, x2, y2] format

    Returns:
        loss: (N,) per-instance CIoU loss
    """
    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0)
    pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    target_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0)
    target_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)

    pred_area = pred_w * pred_h
    target_area = target_w * target_h
    union_area = pred_area + target_area - inter_area + eps

    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    # Distance term
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Aspect ratio term
    with torch.no_grad():
        arctan_pred = torch.atan(pred_w / (pred_h + eps))
        arctan_target = torch.atan(target_w / (target_h + eps))
        v = (4 / (math.pi ** 2)) * (arctan_pred - arctan_target) ** 2

    alpha_ciou = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha_ciou * v

    return 1 - ciou


class ARFScaleAwareLoss(nn.Module):
    """
    ARF-Scale-Aware Loss.

    Reweights the CIoU regression loss by an inverse object area weighting
    function to amplify gradients for small objects.

    L_total = L_cls + L_box + L_dfl + lambda * L_sa
    L_sa = (1/N) * sum_i w(a_i) * L_CIoU(b_hat_i, b_i*)
    w(a_i) = 1 + alpha * exp(-beta * a_i / a_bar)

    Args:
        alpha: Maximum gradient amplification factor (default: 2.0)
        beta: Decay rate controlling the scale sensitivity (default: 3.0)
        lambda_sa: Weight of the scale-aware loss term (default: 0.5)
    """

    def __init__(self, alpha=2.0, beta=3.0, lambda_sa=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_sa = lambda_sa

    def compute_scale_weights(self, target_areas):
        """
        Compute inverse-scale weights for each instance.

        w(a_i) = 1 + alpha * exp(-beta * a_i / a_bar)

        Args:
            target_areas: (N,) areas of target bounding boxes

        Returns:
            weights: (N,) scale-aware weights
        """
        if target_areas.numel() == 0:
            return target_areas

        a_bar = target_areas.mean().clamp(min=1e-6)
        weights = 1.0 + self.alpha * torch.exp(-self.beta * target_areas / a_bar)
        return weights

    def forward(self, pred_boxes, target_boxes, target_areas=None):
        """
        Compute the scale-aware loss.

        Args:
            pred_boxes: (N, 4) predicted bounding boxes [x1, y1, x2, y2]
            target_boxes: (N, 4) ground-truth bounding boxes [x1, y1, x2, y2]
            target_areas: (N,) areas of target boxes; if None, computed from
                          target_boxes

        Returns:
            loss: Scalar scale-aware CIoU loss
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0

        # Compute CIoU loss per instance
        ciou = ciou_loss(pred_boxes, target_boxes)  # (N,)

        # Compute target areas if not provided
        if target_areas is None:
            tw = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0)
            th = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)
            target_areas = tw * th

        # Compute scale-aware weights
        weights = self.compute_scale_weights(target_areas)  # (N,)

        # Weighted mean
        loss_sa = (weights * ciou).mean()

        return self.lambda_sa * loss_sa
