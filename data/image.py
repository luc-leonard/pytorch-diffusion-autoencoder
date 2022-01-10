import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations
import traceback
import torchaudio


class MyImageFolderDataset(Dataset):
    def __init__(
        self,
        root,
        extensions=[".jpg", ".jpeg", ".png"],
        crop_size=None,
        resize=None,
        **ignored,
    ):
        self.files = []
        for extension in extensions:
            self.files.extend(glob.glob(root + "/**/*" + extension))
            self.files.extend(glob.glob(root + "/*" + extension))
        if resize is not None:
            rescaler = albumentations.SmallestMaxSize(max_size=resize)
        else:
            rescaler = albumentations.NoOp()
        if crop_size is not None:
            cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
        else:
            cropper = albumentations.NoOp()
        self.transform = albumentations.Compose([rescaler, cropper])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index]).convert("RGB")
            image = np.array(image).astype(np.uint8)
            image = self.transform(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
            return {
                "image": torch.tensor(image).permute(2, 0, 1),
                "path": self.files[index],
            }
        except Exception as e:
            print(e)
            print(traceback.format_exc())
