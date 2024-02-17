import torch.nn as nn
import torch.nn.functional as F


class DSCLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = -1, alpha: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = 1
        self.alpha = alpha

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=-1)
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes)
        decayed_probs = ((1 - probs) ** self.alpha) * probs
        nom = 2 * decayed_probs * one_hot_labels + self.gamma
        denom = decayed_probs + one_hot_labels + self.gamma
        ignorance_mask = (labels != self.ignore_index)
        loss = (1 - nom / denom).sum(-1) * ignorance_mask
        return loss.sum() / ignorance_mask.sum()
