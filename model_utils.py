from transformers import AutoModelForTokenClassification, AutoConfig
import torch.nn as nn


def create_model(model_name_or_path: str, labels_mapping: dict, max_len:int = 1024):
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path,
                                                            max_position_embeddings=max_len,
                                                            ignore_mismatched_sizes=True)
    model.classifier = nn.Linear(model.config.hidden_size, len(labels_mapping))
    return model
