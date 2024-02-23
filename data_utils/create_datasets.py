import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import PreTrainedTokenizer

from .ner_dataset import NERDataset


class DatasetTokenizer:
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int, labels_mapping: dict):
        self.tokenizer = tokenizer
        self.labels_mapping = labels_mapping
        self.data = data
        self.max_len = max_len

    def re_tokenize_row(self, tokens, labels):
        tokenized_inputs = self.tokenizer(tokens, truncation=True,
                                          is_split_into_words=True, 
                                          add_special_tokens=False,
                                          max_length=self.max_len)

        row_tokens, word_inds = tokenized_inputs['input_ids'], tokenized_inputs.word_ids()

        row_labels = []
        for word_ind in word_inds:
            label = labels[word_ind]
            if label.startswith('B-') or label.startswith('I-'):
                label = label.split('-')[-1]
            label = self.labels_mapping[label]
            row_labels.append(label)

        return [row_tokens, row_labels, word_inds]

    def re_tokenize(self):
        tokens = self.data['tokens']
        labels = self.data['labels']
        processed_rows = []
        for row in range(len(tokens)):
            row_tokens = tokens[row]
            row_labels = labels[row]
            processed_row = self.re_tokenize_row(row_tokens, row_labels)
            processed_rows.append(processed_row)
        return pd.DataFrame(processed_rows, columns=['tokens', 'labels', 'word_inds'], dtype='object')


def create_dataset(paths: list, tokenizer: PreTrainedTokenizer, max_len: int,
                   labels_mapping: dict, force_recreate=False):
    save_file_name = './processed_data.csv'
    if not os.path.exists(save_file_name) or force_recreate:
        print("Start data processing...")
        processed_data = None
        for path in paths:
            raw_data = pd.read_json(path, orient='records')
            raw_data = raw_data[raw_data['labels'].map(lambda x: any([label != 'O' for label in x]))].reset_index(drop=True)
            re_tokenizer = DatasetTokenizer(raw_data, tokenizer, max_len, labels_mapping)
            if processed_data is None:
                processed_data = re_tokenizer.re_tokenize()
            else:
                processed_data = pd.concat([processed_data, re_tokenizer.re_tokenize()], ignore_index=True)
        processed_data.to_csv(save_file_name)
    else:
        print("Found cached data in", save_file_name)
    all_data = pd.read_csv(save_file_name)
    train_data, val_data = train_test_split(all_data, test_size=0.25, random_state=42)
    return NERDataset(train_data.reset_index(drop=True)), NERDataset(val_data.reset_index(drop=True))



