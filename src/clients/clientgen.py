import torch
import numpy as np
import time

from torch import nn

from src.clients.clientbase import Client


class clientGen(Client):
    def __init__(self, args, cid, train_set, **kwargs):
        super().__init__(args, cid, train_set, **kwargs)

        train_loader = self.train_set.get_dataloader(batch_size=self.batch_size)
        for x, y in train_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.embed(x).detach()
            break
        self.feature_dim = rep.shape[1]

        self.sample_per_class = torch.zeros(self.dataset_info['num_classes'])
        train_loader = self.train_set.get_dataloader(batch_size=self.batch_size)
        for x, y in train_loader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1

        self.qualified_labels = []
        self.generative_model = None
        self.localize_feature_extractor = args.localize_feature_extractor
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

                labels = np.random.choice(self.qualified_labels, self.batch_size)
                labels = torch.LongTensor(labels).to(self.device)
                z = self.generative_model(labels)
                loss += self.criterion(self.model.head(z), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_batch.append(loss.item())

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        train_acc = self.train_metrics()
        self.logger.info(f"Client {self.cid} training done, loss: {np.mean(loss_batch):.4f} train acc: {train_acc:.2f}")

    def set_parameters(self, model, generative_model):
        if self.localize_feature_extractor:
            for new_param, old_param in zip(model.parameters(), self.model.classifier.parameters()):
                old_param.data = new_param.data.clone()
        else:
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()

        self.generative_model = generative_model
