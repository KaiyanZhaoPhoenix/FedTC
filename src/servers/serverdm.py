import time

import numpy as np
import swanlab as wandb
import torch
import os.path
import pandas as pd
from torch.optim.lr_scheduler import StepLR

from src.utils.aug_utils import DiffAug
from src.servers.serverbase import Server
from src.clients.clientdm import clientDM
from torch.utils.data import DataLoader, TensorDataset


class FedDM(Server):
    def __init__(self, args, times) -> None:
        super().__init__(args, times)
        self.set_clients(clientDM)
        self.logger = args.logger
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger.info(f"Already initialized {self.num_clients} clients with model, join ratio {self.join_ratio}")

        self.synthetic_images = None
        self.synthetic_labels = None
        self.syn_img_distribution = None
        self.test_acc = []

    def cal_syn_img_distribution(self):
        syn_img_distribution = [self.args.ipc * len(self.clients[idx].classes) for idx in self.selected_clients]
        syn_img_num = sum(syn_img_distribution)

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
            self.logger.info(f"Selected clients: {[idx for idx in self.selected_clients]}")
            for i, idx in enumerate(self.selected_clients):
                self.clients[idx].train()

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
                    self.logger.info(f"Global Model Epoch {epoch}, loss: {epoch_loss}")

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
