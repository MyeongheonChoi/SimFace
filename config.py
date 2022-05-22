import torch
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EDAConfig:
    idx_2_landmark = dict()
    landmarks = ["Nose", 'REye', 'LEye', 'RM', 'LM', 'REar', 'LEar']

    for i, landmark in enumerate(landmarks):
        idx_2_landmark[i] = landmark

    landmark_2_idx = dict(zip(idx_2_landmark.values(), idx_2_landmark.keys()))


class DAMConfig:
    is_gray = False
    imple_id = 2
    past_imple_id = 0
    past_epoch_id = 29

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-1
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 64
    epochs = 30

    dim_emb = 512
    scale = 1e-3
    num_class = 380
    test_size = 30


class DAMEstimateConfig:
    train_cfg = DAMConfig()

    dim_emb = train_cfg.dim_emb
    num_class = train_cfg.num_class
    scale = train_cfg.scale

    test_size = 30
    batch_size = 64
    imple_id = 0
    epoch_id = 29
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ArcConfig:
    is_gray = True
    test_size =30
    train_path = 'data/kface_gray_train.npz'
    test_path = 'data/kface_gray_test.npz'
    save_path = 'models/trained/Arc'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    backbone = 'resnet_face18'
    normalized = True
    batch_size = 64
    num_classes = 400
    use_se = True
    easy_margin = False

    max_epoch = 30
    lr = 1e-1
    lr_step = 6
    lr_decay = 0.95
    weight_decay = 5e-4

    load_model_name = 'resnet_face100'
    if normalized:
        load_path = 'models/trained/Arc/' + load_model_name + '_normalized' + '.pth'
    else:
        load_path = 'models/trained/Arc/' + load_model_name + '_equalized' + '.pth'


parser = argparse.ArgumentParser(description='Trainer for posenet')
parser.add_argument('--arch', default='iresnet100', type=str,
                    help='backbone architechture')
parser.add_argument('--inf_list', default='', type=str,
                    help='the inference list')
parser.add_argument('--feat_list', type=str,
                    help='The save path for saveing features')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--embedding_size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--resume', default='models/trained/MagFace.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
parser.add_argument('--device', default=device, action='store_true', help='Use the CPU.')
parser.add_argument('--dist', default=1, help='use this if model is trained with dist')
parser.add_argument('--is_gray', default=False, help='use this if model is trained with dist')
parser.add_argument('--test_size', default=30, help='use this if model is trained with dist')
mag_args = parser.parse_args()


class SiameseConfig:
    device = device
    batch_size = 32
    is_gray = True


class CosConfig:
    is_gray = True
    test_size = 30
    train_path = 'data/kface_gray_train.npz'
    test_path = 'data/kface_gray_test.npz'
    save_path = 'models/trained/Cos'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    backbone = 'resnet_face18'
    normalized = True
    batch_size = 64
    num_classes = 400
    use_se = True
    easy_margin = False

    max_epoch = 30
    lr = 1e-1
    lr_step = 6
    lr_decay = 0.95
    weight_decay = 5e-4

    load_model_name = 'CosFace100'
    load_path = f'models/trained/Cos/{load_model_name}.pth'


def load_cfg(extractor_type: str):
    if extractor_type == 'Mag':
        return mag_args
    elif extractor_type == 'Arc':
        return ArcConfig()
    elif extractor_type == 'Norm':
        return DAMConfig()
    elif extractor_type == 'Cos':
        return CosConfig()
    else:
        return SiameseConfig()

