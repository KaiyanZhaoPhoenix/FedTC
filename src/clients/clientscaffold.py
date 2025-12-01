import time

import numpy as np
import torch

from src.clients.clientbase import Client
from src.optimizers.fedoptimizer import SCAFFOLDOptimizer


class clientScaffold(Client):
    def __init__(self, args, cid, train_set, **kwargs) -> None:
        super(clientScaffold, self).__init__(args, cid, train_set, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.num_batches = None
        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None

    def train(self):
        train_loader = self.train_set.get_dataloader(self.batch_size)
        # self.model.to(self.device)
        self.model.train()
        optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)

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
                optimizer.step(self.global_c, self.client_c)

                loss_batch.append(loss.item())

        # self.model.cpu()
        self.num_batches = len(train_loader)
        self.update_yc()
        # self.delta_c, self.delta_y = self.delta_yc(max_local_epochs)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        train_acc = self.train_metrics()
        self.logger.info(f"Client {self.cid} training done, loss: {np.mean(loss_batch):.4f} train acc: {train_acc:.2f}")

    def set_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = model

    def update_yc(self):
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1 / self.num_batches / self.epochs / self.learning_rate * (x - yi)

    def delta_yc(self):
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1 / self.num_batches / self.epochs / self.learning_rate * (x - yi))

        return delta_y, delta_c
