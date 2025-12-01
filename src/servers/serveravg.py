import time
import swanlab as wandb

from src.clients.clientavg import clientAVG
from src.servers.serverbase import Server


class FedAvg(Server):
    def __init__(self, args, times) -> None:
        super(FedAvg, self).__init__(args, times)
        self.set_clients(clientAVG)

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




