from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

class ZIFAR10(Dataset):
    def __init__(self, root, train, download):
        self.dataset = CIFAR10(root=root, train=train, download=download, transform=self.__setTransforms())

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

    def __len__(self):
        return len(self.dataset)
    
    def __setTransforms(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
            transforms.RandomHorizontalFlip(p=0.5)
            ])
            
        return transform