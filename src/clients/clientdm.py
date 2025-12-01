import copy
import os
import time
import torch
from src.clients.clientbase import Client
from src.utils.aug_utils import DiffAug
from src.utils.train_utils import get_network, random_pertube, save_images


class clientDM(Client):
    def __init__(self, args, cid, train_set, **kwargs):
        super().__init__(args, cid, train_set)
        # for dataset distillation
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


    def cal_local_syn_distribution(self):
        self.syn_img_num = self.args.ipc * len(self.classes)
        self.syn_img_idx = [0] + [self.args.ipc * i for i in range(1, len(self.classes) + 1)]
        self.logger.info(f"Client {self.cid} synthetic images distribution: {[self.ipc] * len(self.classes)}")

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

        if self.init == 'real':
            for i, c in enumerate(self.classes):
                self.synthetic_images.data[self.syn_img_idx[i]:self.syn_img_idx[i + 1]] = self.train_set.get_images(
                    c, self.syn_img_idx[i + 1] - self.syn_img_idx[i], avg=False
                ).detach().data
        elif self.init == "noise":
            real_images = self.train_set.get_random_images(self.dc_batch_size).detach().data
            self.synthetic_images.requires_grad_(False)
            self.synthetic_images[:, 0, :, :] = (self.synthetic_images[:, 0, :, :] /
                                                 self.synthetic_images[:, 0, :, :].abs().max() *
                                                 real_images[:, 0, :, :].abs().max())
        else:
            raise NotImplementedError
        self.synthetic_images.requires_grad_(True)

    def train(self):
        start_time = time.time()
        self.init_syn_data()
        optimizer_image = torch.optim.SGD([self.synthetic_images], lr=self.image_lr, momentum=self.momentum, weight_decay=self.weight_decay)

        for it in range(self.dc_iterations):
            if self.train_time_cost['num_rounds'] == 0:
                self.model = get_network(self.args.model_str, self.dataset_info)
                sample_model = copy.deepcopy(self.model).to(self.device)
            else:
                # sample w ~ P(w)
                sample_model = random_pertube(self.model, self.rho).to(self.device)

            loss = torch.tensor(0.0).to(self.device)
            for i, c in enumerate(self.classes):
                real_image = self.train_set.get_images(c, self.dc_batch_size).to(self.device)
                synthetic_image = self.synthetic_images[self.syn_img_idx[i]:self.syn_img_idx[i + 1]]

                n = real_image.shape[0]
                if self.dsa_strategy is not None:
                    seed = int(time.time() * 1000) % 100000
                    augment = DiffAug(single=True, batch=False, siamese=True)
                    img_aug = augment(torch.cat([real_image, synthetic_image]), seed=seed)
                    real_feature = sample_model.embed(img_aug[:n]).detach()
                    synthetic_feature = sample_model.embed(img_aug[n:])
                else:
                    real_feature = sample_model.embed(real_image).detach()
                    synthetic_feature = sample_model.embed(synthetic_image)

                real_logit = sample_model.head(real_feature).detach()
                synthetic_logit = sample_model.head(synthetic_feature)

                # DM Loss
                loss += torch.sum((torch.mean(real_feature, dim=0) - torch.mean(synthetic_feature, dim=0)) ** 2)
                loss += torch.sum((torch.mean(real_logit, dim=0) - torch.mean(synthetic_logit, dim=0)) ** 2)

            # update S_k
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
                self.logger.info(f'client {self.cid}, data condensation {it}, total loss = {loss.item()}, '
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
