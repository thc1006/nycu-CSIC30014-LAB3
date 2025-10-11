import torch, torch.nn as nn, torch.nn.functional as F

class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        n = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.eps) * targets + self.eps / n
        return (-targets * log_probs).sum(dim=1).mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss

class ImprovedFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with per-class alpha weights and label smoothing.
    Optimized for handling Bacteria/Virus confusion in chest X-ray classification.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # alpha: [normal, bacteria, virus, COVID-19] = [1.0, 1.5, 2.0, 1.2]
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, target):
        """
        Args:
            logits: [B, C] raw model outputs
            target: [B] class indices (not one-hot)
        """
        num_classes = logits.size(-1)
        device = logits.device

        # Apply label smoothing
        if self.label_smoothing > 0:
            # Convert target to one-hot with smoothing
            with torch.no_grad():
                smooth_target = torch.zeros_like(logits)
                smooth_target.fill_(self.label_smoothing / (num_classes - 1))
                smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(smooth_target * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(logits, target, reduction='none')

        # Compute focal term
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights (alpha)
        if self.alpha is not None:
            # Make sure alpha is on the same device as logits
            if self.alpha.device != device:
                self.alpha = self.alpha.to(device)
            alpha_t = self.alpha[target]
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        return loss.mean()
