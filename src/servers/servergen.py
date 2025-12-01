import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import swanlab as wandb

from src.clients.clientgen import clientGen
from src.servers.serverbase import Server


class FedGen(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientGen)

        self.generative_model = Generative(
            args.noise_dim,
            args.dataset_info['num_classes'],
            args.hidden_dim,
            self.clients[0].feature_dim,
            self.device
        ).to(self.device)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=args.generator_learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.generative_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=args.learning_rate_decay_gamma)
        self.criterion = nn.CrossEntropyLoss()

        self.qualified_labels = []
        for client in self.clients:
            for yy in range(self.args.dataset_info['num_classes']):
                self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])
        for client in self.clients:
            client.qualified_labels = self.qualified_labels

        self.server_epochs = args.server_epochs
        self.localize_feature_extractor = args.localize_feature_extractor
        if self.localize_feature_extractor:
            self.global_model_head = copy.deepcopy(args.model.classifier)

    def fit(self):
        test_acc = self.test_metrics()
        self.logger.info(f"Initial Global Model Test Accuracy: {test_acc}")
        wandb.log({
            "round": 0,
            "round_cost_time": 0,
            "global_model_test_acc": test_acc,
        })

        for cr in range(self.communication_rounds):
            self.logger.info(f"\n-------------Round number: {cr+1}-------------")
            s_t = time.time()
            self.send_models()
            self.select_clients()
            self.logger.info(f"Selected clients: {[idx for idx in self.selected_clients]}")
            for idx in self.selected_clients:
                self.clients[idx].train()

            self.receive_models()
            self.train_generator()
            self.aggregate_parameters()

            self.budget.append(time.time() - s_t)
            self.logger.info(f"Time Cost: {self.budget[-1]}s")

            test_acc = self.test_metrics()
            self.logger.info(f"Global Model Test Accuracy: {test_acc}")

            wandb.log({
                "round": cr + 1,
                "round_cost_time": self.budget[-1],
                "global_model_test_acc": test_acc,
            })

        self.logger.info(f"Total time cost: {sum(self.budget)}s")
        self.logger.info(f"Final Global Model Test Accuracy: {test_acc}")

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model_head, self.generative_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = self.selected_clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for idx in active_clients:
            tot_samples += len(self.clients[idx].train_set)
            self.uploaded_ids.append(self.clients[idx].cid)
            self.uploaded_weights.append(len(self.clients[idx].train_set))
            if self.localize_feature_extractor:
                self.uploaded_models.append(self.clients[idx].model.classifier)
            else:
                self.uploaded_models.append(self.clients[idx].model)
            # try:
            #     client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
            #                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            # except ZeroDivisionError:
            #     client_time_cost = 0
            # if client_time_cost <= self.time_threthold:
            #     tot_samples += client.train_samples
            #     self.uploaded_ids.append(client.id)
            #     self.uploaded_weights.append(client.train_samples)
            #     if self.localize_feature_extractor:
            #         self.uploaded_models.append(client.model.classifier)
            #     else:
            #         self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def train_generator(self):
        self.generative_model.train()

        for _ in range(self.server_epochs):
            labels = np.random.choice(self.qualified_labels, self.batch_size)
            labels = torch.LongTensor(labels).to(self.device)
            z = self.generative_model(labels)

            logits = 0
            for w, model in zip(self.uploaded_weights, self.uploaded_models):
                model.eval()
                if self.localize_feature_extractor:
                    logits += model(z) * w
                else:
                    logits += model.head(z) * w

            self.generative_optimizer.zero_grad()
            loss = self.criterion(logits, labels)
            loss.backward()
            self.generative_optimizer.step()

        self.generative_learning_rate_scheduler.step()


# based on official code https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/trainmodel/generator.py
class Generative(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device)  # sampling from Gaussian

        y_input = F.one_hot(labels, self.num_classes)
        z = torch.cat((eps, y_input), dim=1)

        z = self.fc1(z)
        z = self.fc(z)

        return z