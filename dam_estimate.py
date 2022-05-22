import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DAMEstimateConfig
from models.DAM import Extractor, Sims
from util.dam.data import load_data, SimSet
from util.dam.metric import discrepancy

import torch
import torchvision.models.resnet as resnet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    extractor = Extractor(resnet.Bottleneck, [3, 4, 6, 3], True).to(cfg.device)
    extractor.load_state_dict(torch.load(f"{root_path}/save/model/DAM_extractor_{cfg.imple_id}_E{cfg.epoch_id}.pt"))

    model_dam = Sims(cfg.dim_emb, cfg.num_class, cfg.scale).to(cfg.device)
    model_dam.load_state_dict(torch.load(f"{root_path}/save/model/DAM_dam_{cfg.imple_id}_E{cfg.epoch_id}.pt"))

    transform = transforms.Compose([transforms.ToTensor()])

    # train_loader, test_loader = data_load(transform)
    # train_data, train_emb, train_labels = extract_emb(extractor, model_dam, train_loader)
    # test_data, test_emb, test_labels = extract_emb(extractor, model_dam, test_loader)

    gallery = np.load(f"{root_path}/save/estimates/train.npz")
    newbie = np.load(f"{root_path}/save/estimates/test.npz")

    newbie_emb = newbie['emb']
    newbie_label = newbie['label']

    # Different case
    sample = np.random.choice(np.unique(newbie_label), cfg.test_size*2, replace=False)
    cases = list(zip(sample[:cfg.test_size], sample[cfg.test_size:]))
    different_case = dict()
    print(f"# of Different case : {len(cases)}")
    for idx, case in tqdm(enumerate(cases)):
        photo1 = np.random.choice(np.argwhere(newbie_label == case[0])[:, 0], 1)
        case1 = newbie_emb[photo1]
        case1_label = newbie_label[photo1][0]

        photo2 = np.random.choice(np.argwhere(newbie_label == case[1])[:, 0], 1)
        case2 = newbie_emb[photo2]
        case2_label = newbie_label[photo2][0]

        sim = discrepancy(gallery['emb'], case1, case2, scale=cfg.scale)
        different_case[idx] = [case1_label, case2_label, sim]
    different_result = pd.DataFrame.from_dict(different_case, orient='index')

    # Same case
    sample = np.random.choice(np.unique(newbie_label), cfg.test_size, replace=False)
    same_case = dict()
    print(f"# of Same case : {len(sample)}")
    for idx, case in tqdm(enumerate(sample)):
        target = np.argwhere(newbie_label == case)
        photo1, photo2 = np.random.choice(target[:, 0], 2, replace=False)
        case1 = newbie_emb[[photo1]]
        case1_label = newbie_label[photo1]
        case2 = newbie_emb[[photo2]]
        case2_label = newbie_label[photo2]
        sim = discrepancy(gallery['emb'], case1, case2, scale=cfg.scale)
        same_case[idx] = [case1_label, case2_label, sim]

    same_result = pd.DataFrame.from_dict(same_case, orient='index')

    final = pd.concat((different_result, same_result), axis=0)
    final.to_csv(f"{root_path}/save/estimates/results_{cfg.test_size}.csv")


def extract_emb(extractor, model, loader):
    embs = []
    datum = []
    labels = []
    extractor.eval()
    with torch.no_grad():
        for data, label in tqdm(loader):
            feature = extractor(data)
            emb = model.extract_emb(feature)

            datum.extend(data.detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())
            embs.extend(emb.detach().cpu().numpy())
    embs = np.array(embs)
    datum = np.array(datum)
    labels = np.array(labels)
    return datum, embs, labels


def data_load(transform):
    train_x, train_y, test_x, test_y = load_data(root_path)
    train_set = SimSet(train_x, train_y, transform)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False)
    test_set = SimSet(test_x, test_y, transform)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader



if __name__ == '__main__':
    cfg = DAMEstimateConfig()
    root_path = os.getcwd()
    main()
