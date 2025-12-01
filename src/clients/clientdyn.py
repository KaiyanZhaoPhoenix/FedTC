import copy
import torch
import numpy as np
import time

from torch import nn

from src.clients.clientbase import Client


class clientDyn(Client):
    def __init__(self, args, cid, train_set, **kwargs) -> None:
        super().__init__(args, cid, train_set, **kwargs)

        self.alpha = args.alpha

        self.global_model_vector = None
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        train_loader = self.train_set.get_dataloader(batch_size=self.batch_size)

        start_time = time.time()
        loss_batch = []
        for epoch in range(self.epochs):
            loss_batch = []
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                if self.global_model_vector is not None:
                    v1 = model_parameter_vector(self.model)
                    loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_batch.append(loss.item())

        if self.global_model_vector is not None:
            v1 = model_parameter_vector(self.model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        train_acc = self.train_metrics()
        self.logger.info(f"Client {self.cid} training done, loss: {np.mean(loss_batch):.4f} train acc: {train_acc:.2f}")

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()

        self.global_model_vector = model_parameter_vector(model).detach().clone()


def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)