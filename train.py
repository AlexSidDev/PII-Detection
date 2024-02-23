from data_utils import create_dataset, DataCollator
from model_utils import create_model
from training_utils import Trainer, DSCLoss
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score

from functools import partial
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='dslim/distilbert-NER')
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--acccum_step', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    class_names = ['O', 'NAME_STUDENT', 'EMAIL', 'USERNAME', 'ID_NUM', 'PHONE_NUM', 'URL_PERSONAL', 'STREET_ADDRESS']

    labels_mapping = {class_name: ind for ind, class_name in enumerate(class_names)}

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = create_model(args.model_path, labels_mapping)

    train_dataset, val_dataset = create_dataset(['dataset/train.json', 'dataset/mixtral-8x7b-v1.json'],
                                                max_len=model.config.max_position_embeddings,
                                                tokenizer=tokenizer, labels_mapping=labels_mapping)

    collator = DataCollator(model.config.max_position_embeddings, token_pad_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, drop_last=False)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    #loss = DSCLoss(len(class_names))
    loss = CrossEntropyLoss(ignore_index=-1)
    trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader,
                      partial(f1_score, average='micro'), device=device)
    trainer.train(3, accumulation_step=args.acccum_step)


