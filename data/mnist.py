import torchvision
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import ToTensor


class MNIST(Dataset):
    def __init__(self, root):
        train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True)
        self.ds = ConcatDataset([train_dataset, test_dataset])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds[index]
        return ToTensor()(sample[0]), sample[1]
