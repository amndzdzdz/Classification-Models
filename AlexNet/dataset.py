import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from utils import pca_color_aug

class ZIFAR10(Dataset):
    def __init__(self, root, train, download):
        self.dataset = CIFAR10(root=root, train=train, download=download, transform=self.__setTransforms())

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = pca_color_aug(image)
        return image, label

    def __len__(self):
        return len(self.dataset)
    
    def __setTransforms(self):
        transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.RandomCrop((227, 227))
            ])

        return transform