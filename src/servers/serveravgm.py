import time
import copy
import swanlab as wandb
import torch

from src.clients.clientavg import clientAVG
from src.servers.serverbase import Server


class FedAvgM(Server):
    def __init__(self, args, times) -> None:
        super(FedAvgM, self).__init__(args, times)
        self.set_clients(clientAVG)

        self.momentum_v = [torch.zeros_like(param.data) for param in self.global_model.parameters()]
        self.momentum_factor = args.beta

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # 1. 计算客户端模型的加权平均
        avg_model = copy.deepcopy(self.uploaded_models[0])
        for param in avg_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                avg_param.data += client_param.data * w

        # 2. 计算当前更新量 (delta = 平均模型 - 全局模型)
        delta = [
            avg_param.data - global_param.data
            for avg_param, global_param in zip(avg_model.parameters(), self.global_model.parameters())
        ]

        # 3. 更新动量 (v = momentum * v + delta)
        for i in range(len(self.momentum_v)):
            self.momentum_v[i] = self.momentum_factor * self.momentum_v[i] + delta[i]

        # 4. 使用动量更新全局模型 (global_model += v)
        for i, param in enumerate(self.global_model.parameters()):
            param.data += self.momentum_v[i]

    def fit(self):
        test_acc = self.test_metrics()
        self.logger.info(f"Initial Global Model Test Accuracy: {test_acc}")
        wandb.log({
            "round": 0,
            "round_cost_time": 0,
            "global_model_test_acc": test_acc,
        })

        for cr in range(self.communication_rounds):
            self.logger.info(f"\n-------------Round number: {cr + 1}-------------")
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