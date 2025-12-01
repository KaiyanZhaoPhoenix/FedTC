import copy
import os
import json
import time
from collections import Counter

import torch
import random
from torch.utils.data import Subset
from torchvision.transforms import transforms

from src.clients.clientbase import Client
from torch.utils.data import Dataset, DataLoader
from datasets.utils.dataset_utils import DomainNetDataset, PerLabelDatasetNonIID, get_dataset_info, get_dataset


class Server:
    def __init__(self, args, times) -> None:
        self.args = args
        self.times = times
        self.logger = args.logger

        self.eval_gap = args.eval_gap
        self.global_model = copy.deepcopy(args.model)
        self.batch_size = args.global_batch_size
        self.model_epochs = args.global_model_epochs
        self.learning_rate = args.global_learning_rate
        self.momentum = args.global_momentum
        self.weight_decay = args.global_weight_decay

        self.dataset = args.dataset
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.communication_rounds = args.communication_rounds
        self.device = args.device

        self.save_folder_name = args.save_folder_name

        self.clients = []
        self.budget = []
        self.selected_clients = []
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []


    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        # w_updated = copy.deepcopy(self.uploaded_models[0])
        # for k in w_updated.keys():
        #     w_updated[k] = w_updated[k] * self.uploaded_weights[0]
        #     for i in range(1, len(self.uploaded_models)):
        #         w_updated[k] += self.uploaded_models[i][k] * self.uploaded_weights[i]
        #     w_updated[k] = w_updated[k] / sum(self.uploaded_weights)
        # self.global_model.load_state_dict(copy.deepcopy(w_updated))

    def receive_models(self):
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []

        assert (len(self.selected_clients) > 0)

        total_samples = 0
        for i, idx in enumerate(self.selected_clients):
            self.uploaded_ids.append(self.clients[idx].cid)
            self.uploaded_weights.append(len(self.clients[idx].train_set))
            self.uploaded_models.append(copy.deepcopy(self.clients[idx].model))
            total_samples += len(self.clients[idx].train_set)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / total_samples

    def set_clients(self, clientObj):
        self.clients = []

        if self.args.dataset == 'DomainNet':
            transform_train = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
            ])
            train_sets, client_classes = [], []
            domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
            for domain in domains:
                train_set = DomainNetDataset(None, domain, transform=transform_train)
                counter = Counter(train_set.labels)
                classes = sorted([key for key, value in counter.items() if value > 10])
                train_sets.append(train_set)
                client_classes.append(classes)
        else:
            with open(self.args.split_file, 'r') as file:
                file_data = json.load(file)
            train_set, test_set = get_dataset(self.args.algorithm, self.args.dataset, self.args.dataset_root)
            client_indices, client_classes = file_data['client_idx'], file_data['client_classes']
            train_sets = [Subset(train_set, indices) for indices in client_indices]

        for i in range(self.num_clients):
            self.clients.append(clientObj(self.args, i, train_set=PerLabelDatasetNonIID(
                train_sets[i],
                client_classes[i],
                self.args.dataset_info['channel'],
                self.device,
            )))

    def select_clients(self):
        self.selected_clients = random.sample(range(len(self.clients)), int(round(len(self.clients) * self.join_ratio)))

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)
            # print(id(client.model))
            # print(id(self.global_model), '\n')

            try:
                assert id(client.model) != id(self.global_model)
                assert str(client.model.state_dict()) == str(self.global_model.state_dict())
            except AssertionError:
                self.logger.info(f"AssertionError:\n "
                                 f"client model id: {id(client.model)} global model id: {id(self.global_model)}\n")
                exit()


            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def save_item(self, item, item_name, item_path=None):
        if item_path is None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.dataset == 'DomainNet':
            transforms_test = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
            ])
            domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
            domain_acc = []
            for domain in domains:
                test_set = DomainNetDataset(None, domain, train=False, transform=transforms_test)
                test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in test_loader:
                        images, labels = data
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.global_model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    domain_acc.append(correct / total)
                    self.logger.info(f"Test Accuracy of {domain}: {100 * correct / total}")
            return sum(domain_acc) / len(domain_acc)
        else:
            train_set, test_set = get_dataset(self.args.algorithm, self.dataset, self.args.dataset_root)
            test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

            self.global_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.global_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            return correct / total
