import torch
import numpy as np
import torch.nn as nn
import torchvision.models.resnet as resnet
from config import DAMConfig

cfg = DAMConfig()
conv1x1 = resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock = resnet.BasicBlock


class Extractor(nn.Module):
    def __init__(self, block, layers, zero_init_residual=True):
        super(Extractor, self).__init__()
        self.inplanes = 64

        # inputs = 3x224x224 -> 3x128x128로 바뀜
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)  # 마찬가지로 전부 사이즈 조정
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])  # 3 반복
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 4 반복
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 6 반복
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 3 반복

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):  # planes -> 입력되는 채널 수
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class Sims(nn.Module):
    def __init__(self, dim_emb, num_class, scale, dim_in=2048):
        super(Sims, self).__init__()
        self.layer1 = nn.Linear(dim_in, dim_emb)
        nn.init.xavier_normal(self.layer1.weight)

        self.center = nn.Embedding(num_class, dim_emb)
        nn.init.xavier_normal(self.center.weight)

        self.s = scale
        self.num_class = num_class

    def extract_emb(self, x):
        return self.layer1(x)

    def forward(self, x, y):
        feature_emb = self.layer1(x)
        class_emb = self.center(torch.LongTensor(np.arange(self.num_class)).to(cfg.device))

        sim = self.cosim(feature_emb, class_emb)
        sim = torch.exp(sim)

        result = sim[np.arange(len(x)), y] / sim.sum(dim=1)
        result = - torch.log(result)
        return result

    def cosim(self, v1, v2):
        return self.s * v1@v2.T / v1.norm(p=2, dim=1).unsqueeze(dim=1) / v2.norm(p=2, dim=1).unsqueeze(dim=0)


def dam():
    model_dam = Sims(cfg.dim_emb, cfg.num_class, cfg.scale).to(cfg.device)
    return model_dam
