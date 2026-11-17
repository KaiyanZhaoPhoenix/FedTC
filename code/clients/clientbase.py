import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.holdout import split_indices


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.split_seed = getattr(args, 'split_seed', args.seed)
        self.validation_fraction = getattr(args, 'validation_fraction', 0.)
        self.shared_protocol = getattr(args, 'protocol', 'upstream') == 'shared'
        self.seed = args.seed
        self.round_index = 0
        self._full_train_data = None
        self._test_data = None
        self.train_indices = None
        self.validation_indices = None
        if self.shared_protocol or self.validation_fraction:
            self._prepare_training_partition()

    def _prepare_training_partition(self):
        if self._full_train_data is None:
            self._full_train_data = read_client_data(self.dataset, self.id, is_train=True)
            self.original_train_samples = len(self._full_train_data)
            self.train_indices, self.validation_indices = split_indices(
                self.original_train_samples, self.validation_fraction, self.split_seed, self.id)
            self.train_samples = len(self.train_indices)

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.shared_protocol or self.validation_fraction:
            self._prepare_training_partition()
            train_data = [self._full_train_data[i] for i in self.train_indices]
            generator = torch.Generator().manual_seed(self.seed + 100003 * self.round_index + 97 * self.id)
            return DataLoader(train_data, batch_size, drop_last=True, shuffle=True, generator=generator)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.shared_protocol:
            if self._test_data is None:
                self._test_data = read_client_data(self.dataset, self.id, is_train=False)
            return DataLoader(self._test_data, batch_size, drop_last=False, shuffle=True,
                              generator=torch.Generator().manual_seed(self.seed + self.id))
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
