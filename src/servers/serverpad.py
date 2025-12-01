import time
import swanlab as wandb
import torch
import os.path
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR

from src.utils.aug_utils import DiffAug
from src.servers.serverbase import Server
from src.clients.clientpad import clientPAD
from torch.utils.data import DataLoader, TensorDataset
from src.utils.train_utils import decode_zoom


class FedPAD(Server):
    def __init__(self, args, times) -> None:
        super().__init__(args, times)
        self.uploaded_protos = None
        self.global_protos = None
        self.logger = args.logger
        self.set_clients(clientPAD)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger.info(f"Already initialized {self.num_clients} clients with model, join ratio {self.join_ratio}")

        self.synthetic_images = None
        self.synthetic_labels = None
        self.syn_img_distribution = None
        self.test_acc = []
        self.global_protos = None

        self.factor = args.factor
        if self.factor > 1 and self.args.init == 'mix':
            self.logger.info(
                f"Previous Batch Size: {self.batch_size}, Now Batch Size After divide factor: {self.batch_size // (self.factor ** 2)}")
            self.batch_size = self.batch_size // (self.factor ** 2)

    def cal_syn_img_distribution(self):
        real_img_distribution = np.array([len(self.clients[idx].train_set) for idx in self.selected_clients])
        syn_img_distribution = [self.args.ipc * len(self.clients[idx].classes) for idx in self.selected_clients]
        syn_img_num = sum(syn_img_distribution)
        if not self.args.balance:
            real_img_num = sum(real_img_distribution)
            syn_img_num = sum(syn_img_distribution) if self.args.syn_img_num == -1 else self.args.syn_img_num

            syn_img_distribution = np.floor((real_img_distribution / real_img_num) * syn_img_num).astype(int).tolist()
            syn_img_distribution = [img_num if img_num >= 5 else 5 for img_num in syn_img_distribution]

            differ = sum(syn_img_distribution) - syn_img_num

            if differ >= 0:
                for i in range(differ):
                    idx = syn_img_distribution.index(max(syn_img_distribution))
                    syn_img_distribution[idx] -= 1
            else:
                for i in range(-differ):
                    idx = syn_img_distribution.index(min(syn_img_distribution))
                    syn_img_distribution[idx] += 1

            assert sum(syn_img_distribution) == syn_img_num

        self.syn_img_distribution = syn_img_distribution
        self.logger.info(f"Transmit {syn_img_num} synthetic images current round")
        self.logger.info(f"Selected clients idx: {self.selected_clients}")
        self.logger.info(f"Synthetic images distribution: {np.array(syn_img_distribution)}")

    def receive_syn_images(self, rounds, save_path):
        synthetic_images = []
        synthetic_labels = []
        for cid in self.selected_clients:
            synthetic_images.append(self.clients[cid].synthetic_images)
            synthetic_labels.append(self.clients[cid].synthetic_labels)

        synthetic_images = torch.cat(synthetic_images, dim=0)
        synthetic_labels = torch.cat(synthetic_labels, dim=0)

        self.logger.info(f"Class distribution of synthetic images: {synthetic_labels.type(torch.int64).bincount()}")

        if self.args.single:
            self.synthetic_images = synthetic_images
            self.synthetic_labels = synthetic_labels
        else:
            self.synthetic_images = synthetic_images if rounds == 0 else torch.cat(
                [self.synthetic_images, synthetic_images], dim=0)
            self.synthetic_labels = synthetic_labels if rounds == 0 else torch.cat(
                [self.synthetic_labels, synthetic_labels], dim=0)

        # self.save_item([synthetic_images, synthetic_labels], f"synthetic_images_round_{rounds}", save_path)

    def train_metrics(self):
        self.global_model.eval()
        correct = 0
        total = 0
        train_loader = DataLoader(TensorDataset(self.synthetic_images, self.synthetic_labels),
                                  batch_size=self.batch_size, shuffle=True)
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device).long()
                if self.factor > 1 and self.args.init == 'mix':
                    images, labels = decode_zoom(images, labels, self.factor, self.args.dataset_info['im_size'])
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def fit(self):
        save_path = os.path.join(self.save_folder_name, f"Server")
        test_acc = self.test_metrics()
        self.logger.info(f"Initial Global Model Test Accuracy: {test_acc}")
        wandb.log({
            "round": 0,
            "round_cost_time": 0,
            "global_model_test_acc": test_acc,
        })

        for cr in range(self.communication_rounds):
            self.logger.info(f"\n-------------Round number: {cr+1}-------------")
            """
            Client Model Training
            """
            start_time = time.time()
            self.send_models()
            self.select_clients()
            self.cal_syn_img_distribution()
            self.logger.info(f"Selected clients: {[idx for idx in self.selected_clients]}")

            if cr > 0:
                self.receive_protos()
                self.global_protos = proto_aggregation(self.uploaded_protos)
                self.send_protos()
                self.logger.info(f"Round {cr+1}: Global Prototypes Updated")
                # self.save_item(self.global_protos, f"global_protos_{rounds}", save_path)

            for i, idx in enumerate(self.selected_clients):
                self.clients[idx].train(self.syn_img_distribution[i])

            self.receive_syn_images(cr, save_path)
            self.logger.info(
                f"Round {cr+1}: synthetic images received, global syn images shape: {self.synthetic_images.shape}")

            """
            Global Model Training
            """
            train_loader = DataLoader(TensorDataset(self.synthetic_images, self.synthetic_labels),
                                      batch_size=self.batch_size, shuffle=True)
            # self.save_item(self.global_model.state_dict(), f"global_model_{rounds}", save_path)
            self.global_model.train()
            optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )

            scheduler = StepLR(optimizer, step_size=self.model_epochs // 2, gamma=0.1) if self.args.global_learning_rate_decay else None
            epoch_loss = 0.0

            for epoch in range(self.model_epochs):
                epoch_loss = 0.0
                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device).long()

                    # decode zoom
                    if self.factor > 1 and self.args.init == 'mix':
                        inputs, labels = decode_zoom(inputs, labels, self.factor, self.args.dataset_info['im_size'])

                    # augmentation
                    if self.args.dsa_strategy:
                        augment = DiffAug(single=True, batch=False)
                        inputs = augment(inputs)

                    # backward
                    outputs = self.global_model(inputs)
                    loss = self.criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if self.args.global_learning_rate_decay:
                    scheduler.step()
                if epoch == self.model_epochs / 2 - 1 or epoch == self.model_epochs - 1:
                    self.logger.info(f"Epoch {epoch}, loss: {epoch_loss}")

            self.budget.append(time.time() - start_time)
            self.logger.info(f"Time Cost: {self.budget[-1]}s")

            train_acc = self.train_metrics()
            test_acc = self.test_metrics()
            self.logger.info(
                f"Round {cr+1}, test accuracy: {test_acc * 100:.2f}%, "
                f"train loss: {epoch_loss / len(train_loader):.4f}, train accuracy: {train_acc * 100:.2f}%")

            wandb.log({
                "round:": cr + 1,
                "round_cost_time": self.budget[-1],
                "global_model_train_acc": train_acc,
                "global_model_test_acc": test_acc,
                "global_model_train_loss": epoch_loss / len(train_loader),
            })

            self.test_acc.append(test_acc)

        self.logger.info(f"Total time cost: {sum(self.budget)}s")
        self.logger.info(f"Final Global Model Test Accuracy: {test_acc}")

        # self.save_item(self.global_model.state_dict(), f"final_model", save_path)
        test_acc_path = os.path.join(self.save_folder_name, "test_acc.csv")
        pd.DataFrame(self.test_acc).to_csv(test_acc_path)
        self.logger.info(f"Test Acc List: {self.test_acc}")
        self.logger.info(f"Training finished, test accuracy saved in {test_acc_path}")

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.clients) > 0)

        self.uploaded_protos = []
        for idx in self.selected_clients:
            self.clients[idx].collect_protos()
            self.uploaded_protos.append(self.clients[idx].protos)


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label