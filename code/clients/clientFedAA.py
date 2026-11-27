import copy
import torch
import torch.nn as nn
import numpy as np
import time
from clients.clientbase import Client
# from utils.privacy import *


class clientFedAA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.local_steps = getattr(args, "local_steps", None)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # A step budget counts optimizer updates across successive shuffled epochs.
        budget = self.local_steps
        if budget is not None and not len(trainloader):
            raise ValueError(f'Client {self.id} has no training batch for its step budget.')
        max_local_epochs = self.local_epochs if budget is None else (budget + len(trainloader) - 1) // len(trainloader)
        completed = 0

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()
                completed += 1
                if completed == budget:
                    break
        self.last_train_steps = completed

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

