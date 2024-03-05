import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from datasets import load_dataset


class TinyImageNetDataset(Dataset):
    def __init__(self, split='train'):
        '''
        split: Valid values are ['train', 'valid']
        '''
        self.transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Resize(64),      # Size of a tiny-imagenet image: (64, 64)
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.dataset = load_dataset('Maysee/tiny-imagenet', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]['image']
        image = image.convert("RGB")
        transformed_img = self.transform(image)
        return transformed_img, self.dataset[index]['label']


def image_dataloader(dataset='mnist', batch_size=32):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        trainset = datasets.MNIST('/data/mnist', train=True, download=True, transform=transform)
        testset = datasets.MNIST('/data/mnist', train=False, download=True, transform=transform)

    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),      # Size of a cifar10 image: (32, 32)
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    elif dataset == 'tinyimagenet':
        trainset = TinyImageNetDataset(split='train')
        testset = TinyImageNetDataset(split='valid')

    else:
        raise ValueError("Argument 'dataset' must be one of the values in ['mnist', 'cifar10, 'tinyimagenet'].")

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

