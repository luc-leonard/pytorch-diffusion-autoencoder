import torchvision
from torch.nn import Upsample
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import read_sn3_pascalvincent_tensor
import torch

from model.modules.embeddings import Identity


class InfiniteMNIST(Dataset):
    def __init__(self, path):
        self.data = read_sn3_pascalvincent_tensor(path).long()

    def __getitem__(self, index):
        return self.data[index].unsqueeze(0), torch.tensor(0)

    def __len__(self):
        return self.data.shape[0]


class QMNIST(Dataset):
    def __init__(self, root, size=None):
        self.ds = torchvision.datasets.QMNIST(root=root, what="nist", download=True)
        if size is not None:
            self.transform = torchvision.transforms.Resize(size)
        else:
            self.transform = Identity()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds[index]
        image = self.transform(sample[0])
        return ToTensor()(image), sample[1]


class MNIST(Dataset):
    def __init__(self, root):
        train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True)
        self.ds = ConcatDataset([train_dataset, test_dataset])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds[index]
        image = sample[0]  # .resize((32, 32))
        return ToTensor()(image), sample[1]
