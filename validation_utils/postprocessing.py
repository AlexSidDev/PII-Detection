import pandas as pd
import numpy as np


def postprocess(model_preds: list, word_inds: list, labels_mapping: dict):
    previous_word_ind = None
    previous_label = None
    labels = []
    for token_ind, pred in enumerate(model_preds):
        word_ind = word_inds[token_ind]
        if word_ind == previous_word_ind:
            continue
        prefix = 'I-' if previous_label == pred else 'B-'
        labels.append(prefix + labels_mapping[pred] if pred != 0 else labels_mapping[pred])
        previous_label = pred
        previous_word_ind = word_ind
    return labels