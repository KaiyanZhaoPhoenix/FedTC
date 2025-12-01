import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from torch import nn

from src.clients.clientbase import Client


class clientMOON(Client):
    def __init__(self, args, cid, train_set, **kwargs) -> None:
        super().__init__(args, cid, train_set, **kwargs)

        self.tau = args.tau
        self.mu = args.mu

        self.global_model = None
        self.old_model = copy.deepcopy(self.model)
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
                rep = self.model.embed(data)
                output = self.model.head(rep)
                loss = self.criterion(output, target)

                rep_old = self.old_model.embed(data).detach()
                rep_global = self.global_model.embed(data).detach()
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (
                            torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(
                    F.cosine_similarity(rep, rep_old) / self.tau)))
                loss += self.mu * torch.mean(loss_con)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_batch.append(loss.item())

        # self.model.cpu()
        self.old_model = copy.deepcopy(self.model)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        train_acc = self.train_metrics()
        self.logger.info(f"Client {self.cid} training done, loss: {np.mean(loss_batch):.4f} train acc: {train_acc:.2f}")

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model = copy.deepcopy(model)