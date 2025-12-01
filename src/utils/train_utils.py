import copy
import os
import random
import shutil
import time
import logging
from math import ceil

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from src.models import ConvNet, ResNet18, ResNet18BN
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def sample_random_model(model, rho):
    new_model = copy.deepcopy(model)
    parameters = new_model.parameters()

    mean = parameters.view(-1)
    multivariate_normal = MultivariateNormal(mean, torch.eye(mean.shape[0]))
    distance = rho + 1
    while distance > rho:
        sample = multivariate_normal.sample()
        distance = torch.sqrt(torch.sum((mean - sample) ** 2))

    new_parameters = sample.view(parameters.shape)
    for old_param, new_param in zip(parameters, new_parameters):
        with torch.no_grad():
            old_param.fill_(new_param)

    return new_model


def random_pertube(model, rho):
    new_model = copy.deepcopy(model)
    for p in new_model.parameters():
        gauss = torch.normal(mean=torch.zeros_like(p), std=1)
        if p.grad is None:
            p.grad = gauss
        else:
            p.grad.data.copy_(gauss.data)

    norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in new_model.parameters() if p.grad is not None]), p=2)

    with torch.no_grad():
        scale = rho / (norm + 1e-12)
        scale = torch.clamp(scale, max=1.0)
        for p in new_model.parameters():
            if p.grad is not None:
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)

    return new_model


def get_network(model_name, dataset_info):
    if model_name == "ConvNet":
        model = ConvNet(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes'],
            net_width=128,
            net_depth=3,
            net_act='relu',
            net_norm='instancenorm',
            net_pooling='avgpooling',
            im_size=dataset_info['im_size']
        )
    elif model_name == "ConvNetBN":
        model = ConvNet(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes'],
            net_width=128,
            net_depth=3,
            net_act='relu',
            net_norm='batchnorm',
            net_pooling='avgpooling',
            im_size=dataset_info['im_size']
        )
    elif model_name == "ResNet18":
        model = ResNet18(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes']
        )
    elif model_name == 'ResNet18BN':
        model = ResNet18BN(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes']
        )
    else:
        raise NotImplementedError("only support ConvNet and ResNet")

    return model


def class_wise_accuracy(model, dataloader):
    model.eval()
    correct = {i: 0 for i in range(10)}
    total = {i: 0 for i in range(10)}
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct[label.item()] += 1
                total[label.item()] += 1

    accuracies = {i: correct[i] / total[i] if total[i] > 0 else 0 for i in correct}
    return accuracies


def save_images(num_classes, images, labels, save_path=None):
    all_possible_labels = torch.arange(num_classes)

    unique_labels = labels.unique()
    label_to_indices = {label.item(): (labels == label).nonzero(as_tuple=True)[0].tolist() for label in unique_labels}

    images_per_row = 10

    fig, axs = plt.subplots(num_classes, images_per_row + 1, figsize=(num_classes, num_classes), squeeze=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for row, label in enumerate(all_possible_labels):
        img_num = len(label_to_indices.get(label.item(), []))
        axs[row, 0].text(0.5, 0.5, f'Label {label}\n num: {img_num} ', horizontalalignment='center',
                         verticalalignment='center', fontsize=12, transform=axs[row, 0].transAxes)
        axs[row, 0].axis('off')

        indices = label_to_indices.get(label.item(), [])
        for col in range(1, images_per_row + 1):
            ax = axs[row, col]
            if col - 1 < len(indices):
                img_idx = indices[col - 1]
                img = images[img_idx].cpu().numpy().transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                ax.imshow(img)
            ax.axis('off')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()


def get_logger(save_path, p_flag=False):
    if os.path.exists(save_path) and os.path.isdir(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    log_path = os.path.join(save_path, 'log.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    if p_flag:
        logger.addHandler(console_handler)

    return logger


# For Factor
class Normalize:
    def __init__(self, mean, std, device='cpu'):
        self.mean = torch.tensor(mean, device=device).reshape(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).reshape(1, len(mean), 1, 1)

    def __call__(self, x, seed=-1):
        return (x - self.mean) / self.std


def decode_zoom(img, target, factor, size=(32, 32)):
    """Uniform multi-formation
    """
    resize = nn.Upsample(size=size, mode='bilinear')
    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor ** 2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec