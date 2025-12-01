import copy
import torch
import numpy as np
import time
from src.clients.clientbase import Client


class clientLC(Client):
    def __init__(self, args, cid, train_set, **kwargs):
        super(clientLC, self).__init__(args, cid, train_set, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.sample_per_class = torch.zeros(args.dataset_info["num_classes"]).to(self.device)
        train_loader = self.train_set.get_dataloader(batch_size=self.batch_size)
        for x, y in train_loader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.val = None

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

                calibration = torch.tile(self.val, (data.size(0), 1))
                loss = self.criterion(output - calibration, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_batch.append(loss.item())

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        train_acc = self.train_metrics()
        self.logger.info(f"Client {self.cid} training done, loss: {np.mean(loss_batch):.4f} train acc: {train_acc:.2f}")

    def logits_calibration(self, feat, y):
        logits = self.model.head(feat)
        logits_calibrated = logits - self.calibration
        # print(logits_calibrated)
        # raw = torch.exp(logits_calibrated)
        # print(raw)

        # one_hot = torch.zeros(logits.size(), device=self.device)
        # one_hot.scatter_(1, y.view(-1, 1).long(), 1)

        # numerator = torch.sum(raw * one_hot, dim=1)
        # denominator = torch.sum(raw * (1 - one_hot), dim=1)
        # loss_cal = torch.mean(- torch.log(numerator / denominator))
        # print(loss_cal)
        # input()
        # return loss_cal

