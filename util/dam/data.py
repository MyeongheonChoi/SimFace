import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
from config import DAMConfig


cfg = DAMConfig()


class SimSet(Dataset):
    def __init__(self, data, target, transform=None):
        super(SimSet, self).__init__()
        self.data = data
        self.target = torch.LongTensor(target).to(cfg.device)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(np.uint8(self.data[idx])).convert('RGB')
        img = self.transform(img) if self.transform is not None else img
        img = torch.Tensor(img).to(cfg.device)
        return img, self.target[idx]


def train_val_split(train_x, train_y, ratio=0.1):
    samples = np.random.choice(np.arange(len(train_y)), round(len(train_y)*ratio), replace=False)
    unsamples = [i for i in range(len(train_y)) if i not in samples]
    ftrain_x = train_x[unsamples, :]
    ftrain_y = train_y[unsamples]
    val_x = train_x[samples, :]
    val_y = train_y[samples]

    return ftrain_x, ftrain_y, val_x, val_y


def get_loader(train_set, val_set, test_set):
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def load_dataset(root_path):
    transform = transforms.Compose([transforms.ToTensor()])
    train_x, train_y, test_x, test_y = load_data(root_path)
    train_x, train_y, val_x, val_y, = train_val_split(train_x, train_y)
    train_set = SimSet(train_x, train_y, transform)
    val_set = SimSet(val_x, val_y, transform)
    test_set = SimSet(test_x, test_y, transform)
    return train_set, val_set, test_set


def load_data(root_path, preprocess='equalized'):
    train = np.load(f"{root_path}/data/train_data.npz")
    train_x = train[preprocess]
    oh = train['label'].astype(str)
    train_y = np.ones_like(oh).astype(int)
    for idx, i in enumerate(np.unique(oh)):
        train_y[oh == i] = idx

    test = np.load(f"{root_path}/data/test_data.npz")
    test_x = test[preprocess]
    test_y = test['label']
    return train_x, train_y, test_x, test_y
