import time
import torch
import numpy as np
import copy
from src.clients.clientavg import clientAVG

class clientFedTC(clientAVG):
    def __init__(self, args, cid, train_set, **kwargs) -> None:
        super(clientFedTC, self).__init__(args, cid, train_set, **kwargs)
        self.utility = 0.0
        self.model_delta = None

    def get_loss(self):
        """Helper to compute current loss on training set without updating model"""
        self.model.eval()
        self.model.to(self.device)
        train_loader = self.train_set.get_dataloader(batch_size=self.batch_size)
        loss_sum = 0.0
        samples = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                loss_sum += loss.item() * data.size(0)
                samples += data.size(0)
        return loss_sum / samples if samples > 0 else 0.0

    def train(self):
        # 1. 记录初始状态和 Loss
        initial_model = copy.deepcopy(self.model.state_dict())
        initial_loss = self.get_loss()

        # 2. 正常执行本地训练 (复用 clientAVG 的逻辑，但为了方便计算 delta，我们手动写一下或调用 super)
        # 这里为了确保逻辑清晰，重写训练循环
        self.model.train()
        self.model.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, 
                                    momentum=self.momentum, weight_decay=self.weight_decay)
        train_loader = self.train_set.get_dataloader(batch_size=self.batch_size)

        start_time = time.time()
        for epoch in range(self.epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # 3. 计算 Final Loss 和 Utility (Loss Reduction)
        final_loss = self.get_loss()
        self.utility = initial_loss - final_loss
        
        # 4. 计算 Model Delta (w_k - w_t) 并保存在 CPU 上以节省显存
        self.model_delta = {}
        final_model = self.model.state_dict()
        for key in final_model.keys():
            self.model_delta[key] = (final_model[key] - initial_model[key]).cpu()

        # 打印日志
        # self.logger.info(f"Client {self.cid} utility: {self.utility:.4f}")