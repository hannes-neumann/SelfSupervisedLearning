from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data

class Dataset:
    def __init__(self, root: str, train: bool, transform: transforms.Compose, test_pipeline: bool = False, batch_size: int = 128):
        self.dataset = datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
        if test_pipeline:
            self.dataset = data.Subset(self.dataset, range(200))
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]