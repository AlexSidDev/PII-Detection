import torch
import torch.nn as nn


def to_device(batch: dict, device: str):
    return {k: v.to(device) for k, v in batch.items()}


class Trainer:
    def __init__(self, model, loss, optimizer, train_dataloader, val_dataloader, metric, device: str, scheduler=None):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.metric = metric

    def train(self, epochs: int, val_every: int = 1, accumulation_step: int = 1):
        accumulated_metrics = []
        for epoch in range(epochs):
            for it, inputs in enumerate(self.train_dataloader):
                inputs = to_device(inputs, device)
                labels = inputs.pop('labels')

                outputs = self.model(**inputs)
                logits = outputs.logits
                loss = self.criterion(logits, labels)
                loss.backward()

                if (it + 1) % accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            if (epoch + 1) % val_every == 0:
                model.eval()
                accumulated_metric = 0
                for it, inputs in enumerate(self.val_dataloader):
                    with torch.no_grad():
                        inputs = to_device(inputs, device)
                        labels = inputs.pop('labels')

                        outputs = self.model(**inputs)
                        metric = self.metric(outputs, labels)

                    accumulated_metric += metric / len(self.val_dataloader)
                accumulated_metrics.append(accumulated_metric)
                print(f'Validation. Epoch: {epoch}, {str(metric)}: {accumulated_metric}')
                model.train()
        return accumulated_metrics
