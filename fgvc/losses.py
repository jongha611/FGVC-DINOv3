"""손실 함수 정의와 팩토리."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgvc.config import ExperimentConfig


class FocalLoss(nn.Module):
    """Alpha-Balanced Focal Loss for highly imbalanced multi-class classification.

    `gamma == 0`이면 순수한 클래스 가중치 기반 Weighted Cross-Entropy와 동일하게 동작합니다.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (list or Tensor): Weight for each class.
            gamma (float): Focusing parameter.
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Logits from the model (batch_size, num_classes)
            targets (Tensor): Ground truth labels (batch_size)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Gather alpha weight for each sample's target class
            alpha_weight = self.alpha[targets]
            focal_loss = focal_loss * alpha_weight

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def build_criterion(config: ExperimentConfig, device: torch.device) -> nn.Module:
    """`LOSS` 축에 따라 손실 함수를 생성합니다."""
    if config.loss == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing).to(device)

    if config.loss == "focal":
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma).to(device)

    raise ValueError(f"지원하지 않는 LOSS: {config.loss!r}")
