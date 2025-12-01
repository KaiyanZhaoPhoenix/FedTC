import time

import torch
import numpy as np

from src.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, cid, train_set, **kwargs) -> None:
        super(clientAVG, self).__init__(args, cid, train_set, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def train(self):
        self.model.train()
        self.model.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        train_loader = self.train_set.get_dataloader(batch_size=self.batch_size)

        start_time = time.time()
        loss_batch = []
        for epoch in range(self.epochs):
            loss_batch = []
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_batch.append(loss.item())

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        train_acc = self.train_metrics()
        self.logger.info(f"Client {self.cid} training done, loss: {np.mean(loss_batch):.4f} train acc: {train_acc:.2f}")
