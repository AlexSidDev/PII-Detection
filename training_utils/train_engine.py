import torch
import torch.nn as nn
from tqdm import tqdm

from .train_factory import compute_metrics


def to_device(batch: dict, device: str):
    return {k: v.to(device) for k, v in batch.items()}


def accumulate_metrics(accumulation: dict, new: dict, weight: float):
    for k in accumulation.keys():
        accumulation[k] += new[k] / weight
    return accumulation


def print_metrics(metrics: dict, epoch: int):
    print('Epoch', epoch, end=' ')
    for k in metrics.keys():
        print(k, metrics[k], end=' ')
    print()


class Trainer:
    def __init__(self, model, loss, optimizer, train_dataloader, val_dataloader, device: str, scheduler=None):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler


    def train(self, epochs: int, val_every: int = 1, accumulation_step: int = 1):
        assert epochs % val_every == 0, 'Epochs number should be divisible by \'val_every\' parameter'
        assert accumulation_step > 0, '\'accumulation_step\' parameter should be greater than zero'
        accumulated_metrics = []
        print('Start training')
        best_metric = 0
        for epoch in range(epochs):
            for it, inputs in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc='Training'):
                inputs = to_device(inputs, self.device)
                labels = inputs.pop('labels')

                outputs = self.model(**inputs)
                logits = outputs.logits
                loss = self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1)) / accumulation_step
                loss.backward()

                if (it + 1) % accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()
                
            if (epoch + 1) % val_every == 0:
                self.model.eval()
                accumulated_metric = {'precision': 0, 'recall': 0, 'f1_score': 0, 'fbeta_score': 0}
                for it, inputs in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), desc='Validation'):
                    with torch.no_grad():
                        inputs = to_device(inputs, self.device)
                        labels = inputs.pop('labels')

                        logits = self.model(**inputs).logits
                        preds = torch.argmax(logits, dim=-1).flatten()
                        metrics = compute_metrics(labels.flatten().cpu().numpy(), preds.cpu().numpy())

                    accumulated_metric = accumulate_metrics(accumulated_metric, metrics, len(self.val_dataloader))
                accumulated_metrics.append(accumulated_metric)
                print_metrics(accumulated_metric, epoch=epoch)
                if best_metric < accumulated_metric['fbeta_score']:
                    best_metric = accumulated_metric['fbeta_score']
                    torch.save(self.model, './best_model.pt')
                self.model.train()
        return accumulated_metrics
