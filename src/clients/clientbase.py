import copy
import os
import torch
from datasets.utils.dataset_utils import PerLabelDatasetNonIID

class Client:
    def __init__(self, args, cid, train_set: PerLabelDatasetNonIID, **kwargs) -> None:
        self.args = args
        self.cid = cid
        self.logger = args.logger
        self.train_set = train_set
        self.dataset_info = args.dataset_info
        self.model = copy.deepcopy(args.model)
        self.device = args.device
        self.save_folder_name = args.save_folder_name
        self.classes = list(train_set.indices_class)

        self.epochs = args.local_epochs
        self.learning_rate = args.local_learning_rate
        self.batch_size = args.local_batch_size
        self.momentum = args.local_momentum
        self.weight_decay = args.local_weight_decay

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

    def set_parameters(self, model):
        # self.model = copy.deepcopy(model)
        # use state dict copy model when has batch normalization
        # self.model.load_state_dict(copy.deepcopy(model.state_dict()))
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
    
    def save_item(self, item, item_name, item_path=None):
        if item_path is None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.cid) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path is None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.cid) + "_" + item_name + ".pt"))

    def train_metrics(self):
        self.model.eval()
        train_loader = self.train_set.get_dataloader(batch_size=self.batch_size, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return correct / total

