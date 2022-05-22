import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet

from config import load_cfg
from util.util_data import ProcessedDataset
from util.util_metric import similarity
from models.mag_model.network_inf import builder_inf
from models import Arc, Siamese, DAM, Cos


def main():
    # extractor = load_extractor().to(cfg.device)
    #
    # # Step 0: 전처리
    # # TODO: test 용 사진 받으면 얼굴 인식하여 전처리하거나 일단 112*112로 구성하기
    # # INPUT : 사진 2개
    # # OUTPUT : grayscale 122*112, colorsclae 112*112
    # ##################################################
    # # 여기에 추가하기
    # ##################################################
    # train_x, train_y, test_x, test_y = load_data()
    #
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_set = ProcessedDataset(train_x, train_y, transform, is_gray=cfg.is_gray)
    # test_set = ProcessedDataset(test_x, test_y, transform, is_gray=cfg.is_gray)
    # train_loader, test_loader = get_loader(train_set, test_set)
    #
    # # Step 1 : Extract Embs
    # # Mag: Trainset 5분
    # # Arc : Trainset 1분
    # # DAM : Trainset 2분
    # # Cos : Trainset 1분
    # if extractor_type not in ['Cos', 'Arc']:
    #     file_name = f'save/embs/{dset}/{extractor_type}'
    # else:
    #     file_name = f'save/embs/{dset}/{extractor_type}_{cfg.load_model_name}'
    # train_embs, train_labels = extract_embs(extractor, train_loader)
    # np.savez(f"{file_name}_train.npz", embs=train_embs, labels=train_labels)
    # test_embs, test_labels = extract_embs(extractor, test_loader)
    # np.savez(f"{file_name}_test.npz", embs=test_embs, labels=test_labels)

    # Step 2: Similarity
    if extractor_type not in ['Cos', 'Arc']:
        file_name = f'save/embs/{dset}/{extractor_type}'
    else:
        file_name = f'save/embs/{dset}/{extractor_type}_{cfg.load_model_name}'
    gallery = np.load(f"{file_name}_train.npz")
    newbie = np.load(f"{file_name}_test.npz")
    newbie_emb = newbie['embs']
    newbie_label = newbie['labels']

    # Different case
    sample = np.random.choice(np.unique(newbie_label), cfg.test_size * 2, replace=False)
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

        sim = similarity(metric, case1, case2, gallery['embs'])
        # sim = discrepancy(gallery['emb'], case1, case2, scale=cfg.scale)
        different_case[idx] = [metric, extractor_type, case1_label, case2_label, sim]
    different_result = pd.DataFrame.from_dict(different_case, orient='index')

    # Same case

    sample = np.random.choice(np.unique(newbie_label), cfg.test_size if dset != 'calfw' else cfg.test_size*2, replace=False)
    same_case = dict()
    print(f"# of Same case : {len(sample)}")
    for idx, case in tqdm(enumerate(sample)):
        target = np.argwhere(newbie_label == case)
        try:
            photo1, photo2 = np.random.choice(target[:, 0], 2, replace=False)
        except ValueError:
            continue

        case1 = newbie_emb[[photo1]]
        case1_label = newbie_label[photo1]

        case2 = newbie_emb[[photo2]]
        case2_label = newbie_label[photo2]

        sim = similarity(metric, case1, case2, gallery['embs'])
        # sim = discrepancy(gallery['emb'], case1, case2, scale=cfg.scale)
        same_case[idx] = [metric, extractor_type, case1_label, case2_label, sim]
    same_result = pd.DataFrame.from_dict(same_case, orient='index')

    final = pd.concat((different_result, same_result), axis=0)
    if extractor_type not in ['Cos', 'Arc']:
        file_name = f"{root_path}/save/estimates/{extractor_type}_{metric}.csv"
    else:
        file_name = f"{root_path}/save/estimates/{extractor_type}_{cfg.load_model_name}_{metric}.csv"
    final.to_csv(file_name)


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
        elif cfg.load_model_name == 'resnet_face50':
            model = Arc.resnet_face50()
        elif cfg.load_model_name == 'resnet_face100':
            model = Arc.resnet_face100()

        if cfg.load_model_name == 'resnet_face18':
            new_state_dict = torch.load(cfg.load_path)
        else:
            new_state_dict = OrderedDict()
            for k, v in torch.load(cfg.load_path).items():
                name = k[7:]
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    elif extractor_type == 'Norm':
        model = DAM.Extractor(resnet.Bottleneck, [3, 4, 6, 3], True).to(cfg.device)
        model.load_state_dict(torch.load(f"{root_path}/models/trained/DAM_extractor.pt"))
    elif extractor_type == 'Cos':
        if cfg.load_model_name == 'CosFace18':
            model = Cos.resnet_face18()
        elif cfg.load_model_name == 'CosFace50':
            model = Cos.resnet_face50()
        elif cfg.load_model_name == 'CosFace100':
            model = Cos.resnet_face100()

        new_state_dict = OrderedDict()
        for k, v in torch.load(cfg.load_path).items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
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
    if dset == 'kface_gray':
        train = np.load(f"{root_path}/data/kface_gray_train.npz")
        train_x = train[preprocess]
        oh = train['label'].astype(str)
        train_y = np.ones_like(oh).astype(int)
        for idx, i in enumerate(np.unique(oh)):
            train_y[oh == i] = idx

        test = np.load(f"{root_path}/data/kface_gray_test.npz")
        test_x = test[preprocess]
        test_y = test['label']
    else:
        raw = np.load(f"{root_path}/data/{dset}.npz")
        data = raw[preprocess]

        total_label = pd.Series(raw['path']).apply(lambda x: x.split('/')[-1][4:])
        missing_label = pd.Series(raw['not_preprocessed_path'])

        labels = []
        for label in total_label:
            if label in missing_label.tolist():
                continue
            name = label.split('_')[:-1]
            labels.append('_'.join(name))
        labels = np.array(labels)

        target = np.ones_like(labels).astype(int)
        for idx, i in enumerate(np.unique(labels)):
            target[labels == i] = idx

        train_x = data[target % 10 != 9]
        train_y = target[target % 10 != 9]
        test_x = data[target % 10 == 9]
        test_y = target[target % 10 == 9]

    return train_x, train_y, test_x, test_y


def extract_embs(extractor, loader):
    if extractor_type == 'Norm':
        model_dam = DAM.Sims(cfg.dim_emb, cfg.num_class, cfg.scale).to(cfg.device)
        model_dam.load_state_dict(torch.load(f"{root_path}/models/trained/DAM_dam.pt"))

    embs = []
    labels = []
    extractor.eval()
    with torch.no_grad():
        for data, label in tqdm(loader):
            emb = extractor(data)
            if extractor_type == 'Norm':
                emb = model_dam.extract_emb(emb)

            labels.extend(label.detach().cpu().numpy())
            embs.extend(emb.detach().cpu().numpy())
    embs = np.array(embs)
    labels = np.array(labels)
    return embs, labels


if __name__ == "__main__":
    root_path = os.getcwd()
    cudnn.benchmark = True
    # Finished : 'kface_gray', 'calfw'
    # not yet DB : 'cfp', 'lfw', 'agedb'
    db = ['kface_gray', 'calfw']

    # Finished : Mag, Arc, Norm, Cos
    # Not yet : Siamese,
    extractors = ["Arc", 'Cos', 'Mag', 'Norm']

    # 'DAM', "Cos", 'L2'
    metrics = ['DAM', 'Cos', 'L2']

    for extractor_type, metric, dset in product(extractors, metrics, db):
        print('='*30)
        print(f"Extractor : {extractor_type}, Metric : {metric}, DB: {dset}")
        cfg = load_cfg(extractor_type)
        main()
