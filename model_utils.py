from transformers import AutoModelForTokenClassification, AutoConfig
import torch.nn as nn


def create_model(model_name_or_path: str, labels_mapping: dict):
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    model.classifier = nn.Linear(model.config.hidden_size, len(labels_mapping))
    return model
