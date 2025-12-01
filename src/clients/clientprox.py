import time
import copy

import numpy as np
import torch.nn as nn
from src.optimizers.fedoptimizer import PerturbedGradientDescent
from src.clients.clientbase import Client


class clientProx(Client):
    def __init__(self, args, cid, train_set, **kwargs) -> None:
        super(clientProx, self).__init__(args, cid, train_set, **kwargs)
        self.mu = args.mu
        self.global_params = copy.deepcopy(list(self.model.parameters()))
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.to(self.device)
        self.model.train()
        optimizer = PerturbedGradientDescent(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum ,mu=self.mu, weight_decay=self.weight_decay)
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
                optimizer.step(self.global_params, self.device)

                loss_batch.append(loss.item())

        # self.model.cpu()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        train_acc = self.train_metrics()
        self.logger.info(f"Client {self.cid} training done, loss: {np.mean(loss_batch):.4f} train acc: {train_acc:.2f}")


    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()
