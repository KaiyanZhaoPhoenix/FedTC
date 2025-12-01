import copy
import time
import torch
import swanlab as wandb

from src.clients.clientdyn import clientDyn
from src.servers.serverbase import Server


class FedDyn(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientDyn)

        self.alpha = args.alpha
        self.server_state = copy.deepcopy(args.model)
        for param in self.server_state.parameters():
            param.data = torch.zeros_like(param.data)

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
            self.update_server_state()
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


    def add_parameters(self, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() / self.num_join_clients

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.uploaded_models:
            self.add_parameters(client_model)

        for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
            server_param.data -= (1 / self.alpha) * state_param

    def update_server_state(self):
        assert (len(self.uploaded_models) > 0)

        model_delta = copy.deepcopy(self.uploaded_models[0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.uploaded_models:
            for server_param, client_param, delta_param in zip(self.global_model.parameters(),
                                                               client_model.parameters(), model_delta.parameters()):
                delta_param.data += (client_param - server_param) / self.num_clients

        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= self.alpha * delta_param
