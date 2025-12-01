import os
import copy
import time
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
from src.utils.aug_utils import DiffAug
from src.clients.clientbase import Client
from src.utils.train_utils import get_network, random_pertube, decode_zoom, save_images, Normalize


class clientPAD(Client):
    def __init__(self, args, cid, train_set, **kwargs):
        super().__init__(args, cid, train_set, **kwargs)
        self.lamda = args.lamda
        self.protos = None
        self.global_protos = None
        self.factor = args.factor
        # for synthetic data generation
        self.syn_img_idx = None
        self.syn_img_num = None
        self.ipc = args.ipc
        self.init = args.init
        self.image_lr = args.image_lr
        self.dsa_strategy = args.dsa_strategy
        self.dc_iterations = args.dc_iterations
        self.dc_batch_size = args.dc_batch_size
        self.synthetic_images, self.synthetic_labels = None, None
        # for differential privacy
        self.gnb = args.gnb
        self.rho = args.rho
        self.sigma = args.sigma

    def add_laplace_noise(self, num_classes: list, sensitivity, epsilon):
        for i, c in enumerate(self.classes):
            noise = np.random.laplace(0, sensitivity / epsilon)
            num_classes[i] += noise

        num_classes = np.maximum(5, num_classes)  # 确保噪声后的数量非负
        num_classes = num_classes / np.sum(num_classes)  # 归一化
        noise_num_classes = np.ceil(num_classes * self.syn_img_num).astype(int)

        diff = self.syn_img_num - sum(noise_num_classes)
        while diff != 0:
            for i in range(len(self.classes)):
                if diff == 0:
                    break
                if diff > 0:
                    idx = np.where(noise_num_classes == min(noise_num_classes))[0][0]
                    noise_num_classes[idx] += 1
                    diff -= 1
                else:
                    idx = np.where(noise_num_classes == max(noise_num_classes))[0][0]
                    noise_num_classes[idx] -= 1
                    diff += 1

        return noise_num_classes

    def cal_local_syn_distribution(self):
        if self.args.balance:
            self.syn_img_idx = [0] + [self.args.ipc * i for i in range(1, len(self.classes) + 1)]
            c_img_num = [self.ipc] * len(self.classes)
        else:
            images_distribution = [len(self.train_set.indices_class[c_idx]) for c_idx in
                                   sorted(self.train_set.indices_class)]
            c_img_num = [math.floor(self.syn_img_num * c_img / sum(images_distribution))
                         for c_img in images_distribution]

            c_img_num = self.add_laplace_noise(c_img_num, sensitivity=1, epsilon=1)

        images_idx = 0
        self.syn_img_idx = [0] + [images_idx := images_idx + c_img for c_img in c_img_num]
        self.logger.info(f"Client {self.cid} synthetic images distribution: {c_img_num}")

    def init_syn_data(self):
        self.cal_local_syn_distribution()
        self.synthetic_images = torch.randn(
            size=(
                self.syn_img_num,
                self.dataset_info['channel'],
                self.dataset_info['im_size'][0],
                self.dataset_info['im_size'][1],
            ),
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )

        if self.init == 'real' or self.factor == 1:
            for i, c in enumerate(self.classes):
                self.synthetic_images.data[self.syn_img_idx[i]:self.syn_img_idx[i + 1]] = self.train_set.get_images(
                    c, self.syn_img_idx[i + 1] - self.syn_img_idx[i], avg=False
                ).detach().data
        elif self.init == 'mix':
            self.synthetic_images.data = torch.clamp(self.synthetic_images.data / 4 + 0.5, min=0., max=1.)
            size = self.dataset_info['im_size']
            for idx, c in enumerate(self.classes):
                n = self.syn_img_idx[idx + 1] - self.syn_img_idx[idx]
                real_images = self.train_set.get_images(c, n * self.factor ** 2).detach().data

                s = size[0] // self.factor
                remained = size[0] % self.factor
                k = 0

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(real_images[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.synthetic_images.data[self.syn_img_idx[idx]:self.syn_img_idx[idx + 1], :,
                        h_loc:h_loc + h_r,
                        w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r
        elif self.init == 'noise':
            pass
        else:
            raise NotImplementedError

        self.synthetic_images.requires_grad_(True)

    def train(self, syn_img_num):
        start_time = time.time()
        self.syn_img_num = syn_img_num
        self.init_syn_data()
        optimizer_image = torch.optim.SGD([self.synthetic_images], lr=self.image_lr, momentum=self.momentum, weight_decay=self.weight_decay)


        # sample_model = copy.deepcopy(self.model).to(self.device)
        for it in range(self.dc_iterations):
            if self.train_time_cost['num_rounds'] == 0:
                self.model = get_network(self.args.model_str, self.dataset_info)
                sample_model = copy.deepcopy(self.model).to(self.device)
            else:
                # sample_model = random_pertube(self.model, self.rho).to(self.device)
                sample_model = copy.deepcopy(self.model).to(self.device)

            loss = torch.tensor(0.0).to(self.device)
            proto_loss_sum = 0.0
            for i, c in enumerate(self.classes):
                real_image = self.train_set.get_images(c, self.dc_batch_size).to(self.device)
                synthetic_image = self.synthetic_images[self.syn_img_idx[i]:self.syn_img_idx[i + 1]]
                synthetic_label = torch.ones(self.syn_img_idx[i + 1] - self.syn_img_idx[i], device=self.device) * c

                if self.factor > 1 and self.init == 'mix':
                    synthetic_image, _ = decode_zoom(synthetic_image, synthetic_label, self.factor,
                                                     size=self.dataset_info['im_size'])

                n = real_image.shape[0]
                if self.dsa_strategy is not None:
                    if self.factor > 1 and self.init == 'mix':
                        normalize = Normalize(mean=self.args.dataset_info['mean'], std=self.args.dataset_info['std'],
                                              device=self.device)
                        img_aug = normalize(torch.cat([real_image, synthetic_image]))
                    else:
                        img_aug = torch.cat([real_image, synthetic_image])

                    seed = int(time.time() * 1000) % 100000
                    augment = DiffAug(single=True, batch=False, siamese=True)
                    img_aug = augment(img_aug, seed=seed)
                    real_feature = sample_model.embed(img_aug[:n]).detach()
                    synthetic_feature = sample_model.embed(img_aug[n:])
                else:
                    real_feature = sample_model.embed(real_image).detach()
                    synthetic_feature = sample_model.embed(synthetic_image)

                # MMD Loss
                loss += torch.sum((torch.mean(real_feature, dim=0) - torch.mean(synthetic_feature, dim=0)) ** 2)
                if self.global_protos is not None:
                    # Prototype Regularization Loss
                    mmd_loss = torch.sum((self.global_protos[c] - torch.mean(synthetic_feature, dim=0)) ** 2)
                    proto_loss_sum += mmd_loss
                    loss += (mmd_loss * self.lamda)

            optimizer_image.zero_grad()
            loss.backward()

            if self.init == 'noise' and self.args.sigma != 0:
                for i, c in enumerate(self.classes):
                    grad = self.synthetic_images.grad[self.syn_img_idx[i]:self.syn_img_idx[i + 1]]
                    if grad is not None:
                        scale = max(1, torch.norm(grad) / self.gnb)
                        grad.div_(scale)
                        # add gaussian noise
                        noise = torch.normal(mean=0, std=self.sigma * self.gnb, size=grad.size(), device=self.device)
                        grad.add_(noise / self.dc_batch_size)

            optimizer_image.step()

            if it % 500 == 0 or it == (self.dc_iterations - 1):
                self.logger.info(
                    f'client {self.cid}, data condensation {it}, total loss = {loss.item()}, proto_loss_sum = {proto_loss_sum}, '
                    f'avg loss = {loss.item() / len(self.classes)}')

        # return S_k
        self.synthetic_images = self.synthetic_images.detach().cpu()
        self.synthetic_labels = torch.cat(
            [torch.ones(self.syn_img_idx[i + 1] - self.syn_img_idx[i]) * c for i, c in enumerate(self.classes)])

        self.save_syn_images(denormalize=False if self.args.dataset == 'DomainNet' else True)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def save_syn_images(self, denormalize=True):
        syn_images = copy.deepcopy(self.synthetic_images)
        if denormalize:
            for ch in range(self.dataset_info['channel']):
                syn_images[:, ch] = syn_images[:, ch] * self.dataset_info['std'][ch] + self.dataset_info['mean'][ch]
        syn_images = torch.clamp(syn_images, min=0., max=1.)

        rounds = self.train_time_cost['num_rounds']
        save_path = os.path.join(self.save_folder_name, f"Client {self.cid}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, f'round_{rounds}.png')
        save_images(self.dataset_info['num_classes'], syn_images, self.synthetic_labels, save_path=save_name)

    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def collect_protos(self):
        self.model.eval()
        self.model.to(self.device)
        train_loader = self.train_set.get_dataloader(self.dc_batch_size, shuffle=True)

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.embed(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos