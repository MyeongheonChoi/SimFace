import os
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet

from config import load_cfg
from util.util_data import ProcessedDataset
from models.mag_model.network_inf import builder_inf
from models import Arc, Siamese, DAM


def main():
    extractor = load_extractor().to(cfg.device)
    train_set, test_set = load_dataset()
    train_loader, test_loader = get_loader(train_set, test_set)

    # Step 1 : Extract Embs
    # Mag: Trainset 5분
    # Arc : Trainset 1분
    # DAM : Trainset 2분
    save_embs(extractor, train_loader, 'train')
    # save_embs(extractor, test_loader, 'test')

    # Step 2: Similarity

    print()


def load_extractor():
    model = None
    if extractor_type == 'Mag':
        model = builder_inf(cfg)
        model = torch.nn.DataParallel(model)
        if not cfg.cpu_mode:
            model = model.cuda()
    elif extractor_type == 'Arc':
        if cfg.load_model_name == 'resnet_face18':
            model = Arc.resnet_face18()
            model.load_state_dict(torch.load(cfg.load_path, map_location='cpu'))
        elif cfg.load_model_name == 'resnet_face50':
            model = Arc.resnet_face50()
            model.load_state_dict(torch.load(cfg.load_path))
        elif cfg.load_model_name == 'resnet_face100':
            model = Arc.resnet_face100()
            model.load_state_dict(torch.load(cfg.load_path))
    elif extractor_type == 'Norm':
        model = DAM.Extractor(resnet.Bottleneck, [3, 4, 6, 3], True).to(cfg.device)
        model.load_state_dict(torch.load(f"{root_path}/models/trained/DAM_extractor.pt"))
    elif extractor_type == 'Siamese':
        model = Siamese.siamese_origin()
    return model


def get_loader(train_set, test_set):
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=True)

    return train_loader, test_loader


def load_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    train_x, train_y, test_x, test_y = load_data()
    train_set = ProcessedDataset(train_x, train_y, transform, is_gray=cfg.is_gray)
    test_set = ProcessedDataset(test_x, test_y, transform, is_gray=cfg.is_gray)
    return train_set, test_set


def load_data(preprocess='equalized'):
    train = np.load(f"{root_path}/data/gray_train_data.npz")
    train_x = train[preprocess]
    oh = train['label'].astype(str)
    train_y = np.ones_like(oh).astype(int)
    for idx, i in enumerate(np.unique(oh)):
        train_y[oh == i] = idx

    test = np.load(f"{root_path}/data/gray_test_data.npz")
    test_x = test[preprocess]
    test_y = test['label']
    return train_x, train_y, test_x, test_y


def save_embs(extractor, loader, data_type):
    embs = []
    labels = []
    extractor.eval()
    with torch.no_grad():
        for data, label in tqdm(loader):
            emb = extractor(data)

            labels.extend(label.detach().cpu().numpy())
            embs.extend(emb.detach().cpu().numpy())
    embs = np.array(embs)
    labels = np.array(labels)

    np.savez(f'save/embs/{extractor_type}_{data_type}.npz', embs=embs, labels=labels)


if __name__ == "__main__":
    root_path = os.getcwd()
    cudnn.benchmark = True
    # Mag, Arc, DAM, Siamese
    extractors = ["Norm"]
    for extractor_type in extractors:
        cfg = load_cfg(extractor_type)
        main()
