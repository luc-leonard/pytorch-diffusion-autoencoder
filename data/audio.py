import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.datasets import YESNO, SPEECHCOMMANDS
from torch.nn import functional as F

class WavDataset(Dataset):
    def __init__(self, root):
        self.folder = Path(root)
        self.files = glob.glob(root / "*.wav")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torchaudio.load(self.files[idx]), torch.tensor(0)


class YesNoDataset(Dataset):
    def __init__(self, root):
        self.ds = SPEECHCOMMANDS(root=root, download=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]
        data = data[0][:, :16000]
        data = F.pad(data, (0, 16000 - data.shape[1]), value=0)
        return data, torch.tensor(0)