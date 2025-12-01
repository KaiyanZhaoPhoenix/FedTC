import time

import torch
import torch.nn as nn
import swanlab as wandb

from src.clients.clientlc import clientLC
from src.servers.serverbase import Server


class FedLC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientLC)

        self.feature_dim = list(args.model.classifier.parameters())[0].shape[1]
        args.head = nn.Linear(self.feature_dim, args.dataset_info["num_classes"], bias=False).to(args.device)

        sample_per_class = torch.zeros(args.dataset_info["num_classes"]).to(args.device)
        for client in self.clients:
            for y in range(args.dataset_info["num_classes"]):
                sample_per_class[y] += client.sample_per_class[y]
        val = args.tau * sample_per_class ** (-1/4)
        for client in self.clients:
            client.val = val

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
