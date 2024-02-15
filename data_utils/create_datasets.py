import os
import pandas as pd
import torch
from transformers import PreTrainedTokenizer


class DatasetTokenizer:
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer, labels_mapping: dict):
        self.tokenizer = tokenizer
        self.labels_mapping = labels_mapping
        self.data = data

    def re_tokenize_row(self, tokens, labels):
        row_tokens = []
        row_labels = []
        for tok_ind, token in enumerate(tokens):
            label = labels[tok_ind]
            if label.startswith('B-') or label.startswith('I-'):
                label = label.split('-')[-1]
            label = self.labels_mapping[label]
            ids_list = self.tokenizer.encode(token)
            row_tokens.extend(ids_list)
            row_labels.extend([label] * len(ids_list))
        return [row_tokens, row_labels]

    def re_tokenize(self):
        tokens = self.data['tokens']
        labels = self.data['labels']
        processed_rows = []
        for row in range(len(tokens)):
            row_tokens = tokens[row]
            row_labels = labels[row]
            processed_row = self.re_tokenize_row(row_tokens, row_labels)
            processed_rows.append(processed_row)
        return pd.DataFrame(processed_rows, columns=['tokens', 'labels'])


def create_dataset(paths: list, tokenizer: PreTrainedTokenizer, labels_mapping: dict, force_recreation=False):
    save_file_name = './processed_data.csv'
    if not os.path.exists(save_file_name) or force_recreation:
        print("Start data processing...")
        processed_data = None
        for path in paths:
            raw_data = pd.read_json(path, orient='records')
            re_tokenizer = DatasetTokenizer(raw_data, tokenizer, labels_mapping)
            if processed_data is not None:
                processed_data = re_tokenizer.re_tokenize()
            else:
                processed_data = pd.concat([processed_data, re_tokenizer.re_tokenize()], ignore_index=True)
        processed_data.to_csv(save_file_name)
    else:
        print("Found cached data in", save_file_name)
    return pd.read_csv(save_file_name)



