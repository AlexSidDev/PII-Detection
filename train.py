from data_utils import create_dataset, DataCollator
from model_utils import create_model
from training_utils import Trainer, DSCLoss
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score

from functools import partial


if __name__ == '__main__':
    class_names = ['O', 'NAME_STUDENT', 'EMAIL', 'USERNAME', 'ID_NUM', 'PHONE_NUM', 'URL_PERSONAL', 'STREET_ADDRESS']

    labels_mapping = {class_name: ind for ind, class_name in enumerate(class_names)}

    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('dslim/distilbert-NER')

    model = create_model('dslim/distilbert-NER', labels_mapping)

    train_dataset, val_dataset = create_dataset(['dataset/train.json', 'dataset/mixtral-8x7b-v1.json'],
                                                max_len=model.config.max_position_embeddings,
                                                tokenizer=tokenizer, labels_mapping=labels_mapping, force_recreate=True)

    collator = DataCollator(model.config.max_position_embeddings, token_pad_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collator, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=collator, drop_last=True)

    optimizer = AdamW(model.parameters())
    loss = DSCLoss(len(class_names))
    trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader,
                      partial(f1_score, average='micro'), device=device)
    trainer.train(5)


