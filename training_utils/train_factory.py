import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from functools import partial


def compute_metrics(labels, preds, ignore_index: int = -1, beta: float = 5.0):
    make_average = lambda func: partial(func, average='macro')
    metrics = {'precision': make_average(precision_score), 'recall': make_average(recall_score), 
               'f1_score': make_average(f1_score),
               'fbeta_score': partial(fbeta_score,  average='macro', beta=beta)}
    ignorance_mask = labels != ignore_index
    real_inds = np.nonzero(ignorance_mask)[0]
    results = {}
    for name, func in metrics.items():
        results[name] = func(labels[real_inds], preds[real_inds])
    return results


class DSCLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index = -1, alpha: float = 0.6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = 1e-6
        self.alpha = alpha

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=-1)
        ignorance_mask = labels != self.ignore_index
        one_hot_labels = F.one_hot(labels * ignorance_mask, num_classes=self.num_classes)
        decayed_probs = ((1 - probs) ** self.alpha) * probs
        nom = 2 * decayed_probs * one_hot_labels + self.gamma
        denom = decayed_probs + one_hot_labels + self.gamma
        loss = (1 - nom / denom).sum(-1) * ignorance_mask
        return loss.sum() / ignorance_mask.sum()
