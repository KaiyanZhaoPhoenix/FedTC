import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
# from datasets.utils.tinyimagenet_loader import TinyImageNetDataset

def get_dataset_info(dataset, dataset_root):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        class_names = datasets.FashionMNIST(dataset_root, train=True, download=True).classes
    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        class_names = datasets.CIFAR10(dataset_root, train=True, download=True).classes
    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        class_names = datasets.CIFAR100(dataset_root, train=True, download=True).classes
    elif dataset == 'ImageNette':
        channel = 3
        im_size = (64, 64)
        num_classes = 10
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        class_names = datasets.ImageFolder(root=f'{dataset_root}/imagenette2/train').classes
    # elif dataset == 'TinyImageNet':
    #     channel = 3
    #     im_size = (64, 64)
    #     num_classes = 200
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    #     class_names = None
    elif dataset == 'DomainNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 10
        mean, std, class_names = None, None, None
    else:
        exit(f'unknown dataset: {dataset}')

    dataset_info = {
        'channel': channel,
        'im_size': im_size,
        'num_classes': num_classes,
        'classes_names': class_names,
        'mean': mean,
        'std': std,
    }

    return dataset_info


def get_dataset(algo, dataset, dataset_root):
    dataset_info = get_dataset_info(dataset, dataset_root)

    transform_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=dataset_info['mean'], std=dataset_info['std'])])

    # if algo == "FedPAD":
    #     # get original dataset for factor
    #     transform = transforms.Compose([transforms.ToTensor()])
    # else:
    #     transform = transform_val
    
    transform = transform_val

    if dataset == 'MNIST':
        trainset = datasets.MNIST(dataset_root, train=True, download=True, transform=transform)
        testset = datasets.MNIST(dataset_root, train=False, download=True, transform=transform_val)
    elif dataset == 'FashionMNIST':
        trainset = datasets.FashionMNIST(dataset_root, train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(dataset_root, train=False, download=True, transform=transform_val)
    elif dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(dataset_root, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(dataset_root, train=False, download=True, transform=transform_val)
    elif dataset == 'CIFAR100':
        trainset = datasets.CIFAR100(dataset_root, train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(dataset_root, train=False, download=True, transform=transform_val)
    elif dataset == "ImageNette":
        # for download
        # trainset = datasets.Imagenette(dataset_root, split='train', download=True, transform=transform)
        # testset = datasets.Imagenette(dataset_root, split='val', download=True, transform=transform_val)
        transform = transforms.Compose(
            [transforms.Resize([64, 64]), transforms.ToTensor(), transforms.Normalize(mean=dataset_info['mean'], std=dataset_info['std'])]
        )
        trainset = datasets.ImageFolder(root=f'{dataset_root}/imagenette2/train', transform=transform)
        testset = datasets.ImageFolder(root=f'{dataset_root}/imagenette2/val', transform=transform)
    # elif dataset == 'TinyImageNet':
    #     dataset_root += '/tiny-imagenet-200-npy/'
    #     trainset = TinyImageNetDataset(dataset_root, mode='train', transform=transform)
    #     testset = TinyImageNetDataset(dataset_root, mode='val', transform=transform_val, annotations_file=dataset_root + '/val/val_annotations.txt')
    else:
        exit(f'unknown dataset: {dataset}')

    return trainset, testset


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./datasets/data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./datasets/data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)

        label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
                      'windmill': 7, 'wine_glass': 8, 'zebra': 9}

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else './datasets/data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class PerLabelDatasetNonIID:
    def __init__(self, dst_train, classes, channel, device):
        self.images_all = []
        labels_all = []
        self.indices_class = {c: [] for c in classes}

        self.images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            if lab not in classes:
                continue
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(device)
        self.labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

    def __len__(self):
        return self.images_all.shape[0]

    def get_random_images(self, n):
        idx_shuffle = np.random.permutation(range(self.images_all.shape[0]))[:n]
        return self.images_all[idx_shuffle]

    def get_images(self, c, n, avg=False):
        if not avg:
            if len(self.indices_class[c]) >= n:
                idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            else:
                sampled_idx = np.random.choice(self.indices_class[c], n - len(self.indices_class[c]), replace=True)
                idx_shuffle = np.concatenate((self.indices_class[c], sampled_idx), axis=None)
            return self.images_all[idx_shuffle]
        else:
            sampled_imgs = []
            for _ in range(n):
                if len(self.indices_class[c]) >= 5:
                    idx = np.random.choice(self.indices_class[c], 5, replace=False)
                else:
                    idx = np.random.choice(self.indices_class[c], 5, replace=True)
                sampled_imgs.append(torch.mean(self.images_all[idx], dim=0, keepdim=True))
            sampled_imgs = torch.cat(sampled_imgs, dim=0).cuda()
            return sampled_imgs

    def get_dataloader(self, batch_size, shuffle=True, drop_last=False):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.images_all, self.labels_all),
                                           batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
