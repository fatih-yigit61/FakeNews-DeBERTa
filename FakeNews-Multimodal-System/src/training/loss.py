from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import LABEL_SMOOTH_EPS


class MultiTaskLoss(nn.Module):
    """Multi-task loss with per-sample masking for heterogeneous datasets.

    Datasets that lack fake or sentiment labels (e.g. SemEval propaganda) should
    set those target fields to -1.  Any sample with label == -1 is excluded from
    the corresponding loss term so the head is not trained on meaningless signal.

    Manipulation loss uses AsymmetricBinaryLoss when ``use_focal_loss=True``
    (default) — superior to symmetric Focal Loss for imbalanced propaganda
    detection (~21% positive) via separate gamma_pos/gamma_neg control.
    Lambda values balance gradient budget across heads:
      - lambda_fake=1.5  (explicit weight; prevents gradient starvation from other heads)
      - lambda_sentiment=1.3  (boosted for 75% target; sentiment gets ~23% of batch)
      - lambda_manipulation=1.5  (balanced with fake; gets ~30% of batch)
    Note: keep these defaults in sync with TrainerConfig in configs/config.py.
    """

    def __init__(
        self,
        lambda_sentiment: float = 0.5,
        lambda_manipulation: float = 1.0,
        lambda_fake: float = 1.0,
        label_smoothing: float = LABEL_SMOOTH_EPS,
        use_focal_loss: bool = True,
    ):
        super().__init__()
        self.lambda_fake = lambda_fake
        self.lambda_sentiment = lambda_sentiment
        self.lambda_manipulation = lambda_manipulation
        self.label_smoothing = label_smoothing
        self.fake_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # sent_weights registered as buffer so it follows .to(device) automatically.
        # Symmetric [1.0, 2.5, 2.5] boosts both neutral AND positive classes:
        # - Neutral recall was ~55% (under-predicted)
        # - Positive recall was ~65% (also under-predicted vs negative ~80%+)
        # - Symmetric boost prevents the model from defaulting to negative.
        self.register_buffer("sent_weights", torch.tensor([1.0, 2.5, 2.5]))
        # BCEWithLogitsLoss applies sigmoid internally via log-sum-exp trick, avoiding the
        # gradient saturation that occurs when Sigmoid + MSELoss are combined near 0/1.
        self.sentiment_intensity_loss = nn.BCEWithLogitsLoss()
        # Asymmetric Loss (ASL, Ridnik et al. 2021) for manipulation head:
        # - Separately controls positive/negative focusing via gamma_pos/gamma_neg
        # - gamma_neg=3 aggressively down-weights easy negatives (majority class ~79%)
        # - gamma_pos=1 preserves gradient flow for hard positives (minority ~21%)
        # - clip=0.05 probability shifting: shifts negative probabilities, reducing
        #   contribution of very easy negatives even further
        # Superior to symmetric Focal Loss for imbalanced binary classification.
        self.manipulation_loss = AsymmetricBinaryLoss(gamma_pos=1, gamma_neg=3, clip=0.05) if use_focal_loss else nn.BCEWithLogitsLoss()

    @staticmethod
    def _zero(ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(())

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Masks: label == -1 means the dataset does not provide that annotation.
        # All three tasks must be masked — BCEWithLogitsLoss/FocalLoss expect targets
        # in [0, 1]; passing -1.0 produces undefined gradients.
        fake_mask  = targets["fake_label"] != -1
        sent_mask  = targets["sentiment_label"] != -1
        manip_mask = targets["manipulation_label"] != -1

        if fake_mask.any():
            loss_fake = self.fake_loss(predictions["fake_logits"][fake_mask], targets["fake_label"][fake_mask])
        else:
            loss_fake = self._zero(predictions["fake_logits"])

        if sent_mask.any():
            loss_sent_class = F.cross_entropy(
                predictions["sentiment_logits"][sent_mask],
                targets["sentiment_label"][sent_mask],
                weight=self.sent_weights.to(predictions["sentiment_logits"].device),
                label_smoothing=self.label_smoothing,
            )
            # sentiment_intensity targets are derived deterministically from the class label
            # (negative→0.1, neutral→0.5, positive→0.9), so the intensity head encodes
            # no additional information beyond the class head.  Including its loss wastes
            # gradient budget that would otherwise go to manipulation/fake heads.
            # The intensity head is kept in the architecture for inference (forward pass)
            # but excluded from the training loss.
            loss_sent_intensity = self._zero(predictions["sentiment_intensity"])
        else:
            loss_sent_class = self._zero(predictions["sentiment_logits"])
            loss_sent_intensity = self._zero(predictions["sentiment_intensity"])

        if manip_mask.any():
            loss_manipulation = self.manipulation_loss(
                predictions["manipulation_logits"][manip_mask], targets["manipulation_label"][manip_mask]
            )
        else:
            loss_manipulation = self._zero(predictions["manipulation_logits"])
        loss_sentiment = loss_sent_class + loss_sent_intensity
        total_loss = self.lambda_fake * loss_fake + self.lambda_sentiment * loss_sentiment + self.lambda_manipulation * loss_manipulation
        return {
            "total_loss": total_loss,
            "fake_loss": loss_fake,
            "sentiment_class_loss": loss_sent_class,
            "sentiment_intensity_loss": loss_sent_intensity,
            "manipulation_loss": loss_manipulation,
        }


class AsymmetricBinaryLoss(nn.Module):
    """Asymmetric Loss for binary classification (Ridnik et al., 2021).

    Unlike symmetric Focal Loss, ASL uses separate gamma values for positive
    and negative samples.  This is critical for imbalanced propaganda detection
    (~21% positive): gamma_neg >> gamma_pos aggressively down-weights easy
    negatives while preserving gradient flow for hard positives.

    The ``clip`` parameter implements probability shifting: negative sample
    probabilities are shifted by ``clip`` before loss computation, effectively
    discarding the contribution of very easy negatives (p < clip).
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 3.0,
                 clip: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Probability shifting for negatives: clamp p_neg away from 0
        probs_neg = (probs + self.clip).clamp(max=1.0)

        # Separate positive and negative log-probabilities
        log_p_pos = torch.log(probs.clamp(min=1e-8))        # log(p)  for positives
        log_p_neg = torch.log((1.0 - probs_neg).clamp(min=1e-8))  # log(1-p') for negatives

        # Focal modulation
        pos_term = targets * ((1.0 - probs) ** self.gamma_pos) * log_p_pos
        neg_term = (1.0 - targets) * (probs_neg ** self.gamma_neg) * log_p_neg

        loss = -(pos_term + neg_term)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
