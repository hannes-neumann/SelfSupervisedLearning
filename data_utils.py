from torch.utils.data import Dataset

class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x0 = self.transform(x)
            x1 = self.transform(x)
        else:
            x0 = x
            x1 = x
        return (x0, x1), y
    def __len__(self):
        return len(self.subset)

class StandardTransform(Dataset):
    
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)