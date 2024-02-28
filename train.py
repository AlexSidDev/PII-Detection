from data_utils import create_dataset, DataCollator
from model_utils import create_model
from training_utils import Trainer, DSCLoss
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

import argparse


torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    # model params
    parser.add_argument('--model_path', type=str, default='dslim/distilbert-NER',
    help='Path to pre-trained model. Can be link to model from Huggingface hub or to local folder in Transformers format.')
    parser.add_argument('--max_len', type=int, default=1024,
    help='Maximum lenght of sequence that can be passed throught model.')
    # data path
    parser.add_argument('--dataset_paths', nargs='+', type=str,
    help='Path or paths to training datasets in json format.')
    # batch size
    parser.add_argument('--batch_size', type=int, default=1,
    help='Train batch size. It is recommended to use 1 because of huge class disbalance')
    parser.add_argument('--val_batch_size', type=int, default=64)
    # training params
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--acccum_step', type=int, default=1)
    # scheduler
    parser.add_argument('--enable_sched', type=bool, default=True)
    parser.add_argument('--warmup_steps', type=int, default=0)
    #device
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    class_names = ['O', 'NAME_STUDENT', 'EMAIL', 'USERNAME', 'ID_NUM', 'PHONE_NUM', 'URL_PERSONAL', 'STREET_ADDRESS']

    labels_mapping = {class_name: ind for ind, class_name in enumerate(class_names)}

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = create_model(args.model_path, labels_mapping)

    train_dataset, val_dataset = create_dataset(args.dataset_paths,
                                                max_len=model.config.max_position_embeddings,
                                                tokenizer=tokenizer, labels_mapping=labels_mapping,
                                                val_mode='train')

    collator = DataCollator(model.config.max_position_embeddings, token_pad_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collator, drop_last=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.enable_sched:
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps,
                                                     args.epochs * len(train_dataloader) // args.acccum_step)
    else:
        scheduler = None
    loss = DSCLoss(len(class_names))
    #loss = CrossEntropyLoss(ignore_index=-1)
    trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader, device=device, scheduler=scheduler)
    trainer.train(args.epochs, accumulation_step=args.acccum_step)


