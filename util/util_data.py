import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProcessedDataset(Dataset):
    def __init__(self, data, target, transform=None, is_gray=False):
        super(ProcessedDataset, self).__init__()
        self.data = data
        self.target = torch.LongTensor(target).to(device)
        self.transform = transform
        self.is_gray= is_gray

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(np.uint8(self.data[idx]))
        if not self.is_gray:
            img = img.convert('RGB')
        img = self.transform(img) if self.transform is not None else img
        img = torch.Tensor(img).to(device)
        return img, self.target[idx]
