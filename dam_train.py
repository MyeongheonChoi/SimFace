import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.optim import SGD
import torch.backends.cudnn as cudnn
import torchvision.models.resnet as resnet
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import DAMConfig
from models.DAM import Extractor, Sims
from util.dam.data import get_loader, load_dataset


def main():
    # init_params = load_init_params()
    extractor = Extractor(resnet.Bottleneck, [3, 4, 6, 3], True).to(cfg.device)
    # extractor.load_state_dict(torch.load(f"{root_path}/save/model/DAM_extractor_{cfg.past_imple_id}_E{cfg.past_epoch_id}.pt"))

    model_dam = Sims(cfg.dim_emb, cfg.num_class, cfg.scale).to(cfg.device)
    # model_dam.load_state_dict(torch.load(f"{root_path}/save/model/DAM_dam_{cfg.past_imple_id}_E{cfg.past_epoch_id}.pt"))

    optimizer = SGD(extractor.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    cudnn.benchmark = True
    train_set, val_set, test_set = load_dataset(root_path)
    train_loader, val_loader, test_loader = get_loader(train_set, val_set, test_set)

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    print(f"Iterations : {len(train_loader)}")
    for epoch in range(cfg.epochs):
        train_loss = train(extractor, model_dam, train_loader, cfg.device, optimizer)
        train_loss_per_epoch.append(train_loss)

        val_loss = validate(extractor, model_dam, val_loader, cfg.device)
        val_loss_per_epoch.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch} / Train Loss: {train_loss} / Val Loss: {val_loss}")
            torch.save(extractor.state_dict(), f'save/model/DAM_extractor_{cfg.imple_id}_E{epoch}.pt')
            torch.save(model_dam.state_dict(), f'save/model/DAM_dam_{cfg.imple_id}_E{epoch}.pt')

    losses = {"Train": train_loss_per_epoch, "Val": val_loss_per_epoch}
    pd.DataFrame(losses).to_csv(f'save/loss_log/loss_log_{cfg.imple_id}.csv')
    torch.save(extractor.state_dict(), f'save/model/DAM_extractor_{cfg.imple_id}_E{epoch}.pt')
    torch.save(model_dam.state_dict(), f'save/model/DAM_dam_{cfg.imple_id}_E{epoch}.pt')


def load_init_params():
    origin = resnet.resnet50(pretrained=True)
    origin_params = origin.state_dict()
    param_inter = {k: v for k, v in origin_params.items() if k[:2] != 'fc'}
    param_inter = OrderedDict(sorted(param_inter.items(), key=lambda t: t[0]))
    return param_inter


def train(extractor, model, loader, device, optimizer):
    extractor.train()
    model.train()
    loss_per_iter = 0
    for i, (data, label) in tqdm(enumerate(loader)):
        if i > 155:
            continue
        data = data.to(device)
        label = label.to(device)

        facial_feature = extractor(data)
        output = model(facial_feature, label)
        loss = torch.sum(output) * len(label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_per_iter += loss.detach().cpu().numpy()
    return loss_per_iter / len(loader)


def validate(extractor, model, loader, device):
    extractor.eval()
    model.eval()
    val_loss_per_iter = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)

            facial_feature = extractor(data)
            output = model(facial_feature, label)
            loss = torch.sum(output) * len(label)
            val_loss_per_iter += loss.detach().cpu().numpy()

    return val_loss_per_iter / len(loader)


if __name__ == '__main__':
    root_path = os.getcwd()
    cfg = DAMConfig()
    main()
