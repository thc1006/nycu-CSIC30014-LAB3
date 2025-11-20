# -*- coding: utf-8 -*-
"""
Medical-driven loss functions for COVID-19 detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MedicalFocalLoss(nn.Module):
    """
    Improved Focal Loss with medical-specific class weights

    医学驱动的焦点损失，针对胸部X光肺炎分类优化
    """

    def __init__(self, alpha=None, gamma=None, label_smoothing=0.0, reduction='mean'):
        """
        Args:
            alpha: List of class weights [Normal, Bacteria, Virus, COVID-19]
                   Default: [1.0, 1.5, 1.8, 3.0]
            gamma: List of focusing parameters per class
                   Default: [2.0, 2.0, 2.5, 3.0]
            label_smoothing: Label smoothing epsilon (0.0 - no smoothing)
            reduction: 'mean' or 'sum'
        """
        super().__init__()

        # Medical-specific alpha values
        # COVID-19 has highest weight due to severe class imbalance and clinical importance
        if alpha is None:
            alpha = [1.0, 1.5, 1.8, 3.0]  # [Normal, Bacteria, Virus, COVID-19]

        if gamma is None:
            gamma = [2.0, 2.0, 2.5, 3.0]  # [Normal, Bacteria, Virus, COVID-19]

        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size] (class indices) or [batch_size, num_classes] (one-hot)

        Returns:
            loss: scalar or [batch_size] depending on reduction
        """
        # Convert one-hot to class indices if needed
        if targets.dim() == 2:
            targets_onehot = targets
            targets_class = targets.argmax(dim=1)
        else:
            targets_class = targets
            targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).float()

        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = logits.shape[1]
            targets_onehot = targets_onehot * (1 - self.label_smoothing) + \
                             (self.label_smoothing / num_classes)

        # Compute softmax probabilities
        p = torch.softmax(logits, dim=1)

        # Compute cross entropy
        log_p = torch.log_softmax(logits, dim=1)
        ce_loss = -(targets_onehot * log_p).sum(dim=1)

        # Get probability of true class
        p_t = (p * targets_onehot).sum(dim=1)

        # Get class-specific gamma
        gamma_t = self.gamma[targets_class]

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** gamma_t

        # Get class-specific alpha
        alpha_t = self.alpha[targets_class]

        # Compute focal loss
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class COVIDWeightedFocalLoss(nn.Module):
    """
    特别针对COVID-19的加权焦点损失
    动态调整权重基于样本特征
    """

    def __init__(self, base_alpha=None, base_gamma=None, label_smoothing=0.0):
        super().__init__()

        if base_alpha is None:
            base_alpha = [1.0, 1.5, 1.8, 3.0]

        if base_gamma is None:
            base_gamma = [2.0, 2.0, 2.5, 3.0]

        self.base_alpha = torch.tensor(base_alpha, dtype=torch.float32)
        self.base_gamma = torch.tensor(base_gamma, dtype=torch.float32)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, sample_weights=None):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size] (class indices)
            sample_weights: [batch_size] optional per-sample weights
                           特别用于加强COVID-19样本权重

        Returns:
            loss: scalar
        """
        batch_size = logits.shape[0]

        # Create one-hot targets
        targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).float()

        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = logits.shape[1]
            targets_onehot = targets_onehot * (1 - self.label_smoothing) + \
                             (self.label_smoothing / num_classes)

        # Compute probabilities
        p = torch.softmax(logits, dim=1)
        log_p = torch.log_softmax(logits, dim=1)

        # Cross entropy
        ce_loss = -(targets_onehot * log_p).sum(dim=1)

        # Probability of true class
        p_t = (p * targets_onehot).sum(dim=1)

        # Class-specific gamma
        gamma_t = self.base_gamma[targets]

        # Focal weight
        focal_weight = (1 - p_t) ** gamma_t

        # Class-specific alpha
        alpha_t = self.base_alpha[targets]

        # Combine losses
        loss = alpha_t * focal_weight * ce_loss

        # Apply per-sample weights if provided
        if sample_weights is not None:
            loss = loss * sample_weights

        return loss.mean()


class CovidAwareFocalLoss(nn.Module):
    """
    COVID-19感知的焦点损失
    在Virus和COVID-19之间创建梯度
    """

    def __init__(self, alpha=None, gamma=None, label_smoothing=0.05):
        super().__init__()

        # 特别设计用于Virus vs COVID-19区分
        if alpha is None:
            # Normal, Bacteria, Virus, COVID-19
            alpha = [1.0, 1.5, 2.0, 4.0]  # COVID-19权重最高

        if gamma is None:
            gamma = [2.0, 2.0, 3.0, 4.0]  # COVID-19最强聚焦

        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Compute COVID-19 aware focal loss
        """
        # Convert one-hot to indices if needed
        if targets.dim() == 2:
            targets_onehot = targets
            targets_class = targets.argmax(dim=1)
        else:
            targets_class = targets
            targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).float()

        # Label smoothing
        if self.label_smoothing > 0:
            num_classes = logits.shape[1]
            targets_onehot = targets_onehot * (1 - self.label_smoothing) + \
                             (self.label_smoothing / num_classes)

        # Softmax
        p = torch.softmax(logits, dim=1)
        log_p = torch.log_softmax(logits, dim=1)

        # Cross entropy
        ce = -(targets_onehot * log_p).sum(dim=1)

        # Probability of true class
        p_t = (p * targets_onehot).sum(dim=1)

        # Class-specific gamma
        gamma = self.gamma[targets_class]

        # Focal weight
        focal_weight = (1 - p_t) ** gamma

        # Class-specific alpha
        alpha = self.alpha[targets_class]

        # Loss
        loss = alpha * focal_weight * ce

        return loss.mean()


class BalancedMultiLabelLoss(nn.Module):
    """
    平衡的多分类损失，特别考虑类别不平衡
    """

    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            class_weights = [1.0, 1.5, 1.8, 3.0]

        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size] or [batch_size, num_classes]
        """
        # Cross entropy with weights
        loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='mean')
        return loss


def get_loss_function(loss_name='medical_focal', num_classes=4, **kwargs):
    """
    Factory function to get loss function
    """
    if loss_name == 'medical_focal':
        return MedicalFocalLoss(**kwargs)
    elif loss_name == 'covid_weighted_focal':
        return COVIDWeightedFocalLoss(**kwargs)
    elif loss_name == 'covid_aware_focal':
        return CovidAwareFocalLoss(**kwargs)
    elif loss_name == 'balanced_multilabel':
        return BalancedMultiLabelLoss(**kwargs)
    elif loss_name == 'ce':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
