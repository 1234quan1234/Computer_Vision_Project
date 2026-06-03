import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-Entropy loss with label smoothing for vehicle ReID classification.

    Instead of hard one-hot labels, distributes a small probability ``eps``
    uniformly across all classes. This prevents the model from becoming
    overconfident on training identities and improves generalization to
    unseen vehicle IDs during retrieval.

    Loss formula:
        ``L = (1 - eps) * NLL(target) + eps * mean(NLL(all_classes))``

    Args:
        eps: Smoothing factor (default: 0.1). A value of 0.0 reduces to
            standard Cross-Entropy.
    """

    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.

        Args:
            logits: Raw classification scores of shape ``(B, num_classes)``.
            targets: Ground-truth class indices of shape ``(B,)``.

        Returns:
            Scalar loss value (mean over batch).
        """
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=1)
        loss = (1.0 - self.eps) * nll + self.eps * smooth
        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for metric learning in ReID.

    Pulls together all samples sharing the same identity while pushing
    apart samples from different identities. Unlike Triplet Loss which
    only considers one positive and one negative per anchor, SupCon
    leverages *all* positives and negatives in the batch simultaneously,
    leading to more stable gradients and better convergence.

    With PK sampling (P=16 identities, K=8 images each), every anchor
    has K-1=7 positive pairs, providing rich supervisory signal.

    Reference:
        Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

    Args:
        temperature: Scaling factor (tau) for the similarity logits.
            Lower values sharpen the distribution (default: 0.07).
        base_temperature: Normalization constant (typically equal to
            ``temperature``).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss over a batch.

        Args:
            features: L2-normalized feature vectors of shape ``(B, D)``
                or ``(B, 1, D)``.
            labels: Identity labels of shape ``(B,)``.

        Returns:
            Scalar loss value (mean over all anchors).
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        features = F.normalize(features, dim=-1)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = features.shape[1]

        logits = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max = torch.max(logits, dim=1, keepdim=True).values
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
