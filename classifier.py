import torch
import torch.nn as nn
# from data import classes


def Classifier(cls_num, output_size, backbone_type='alexnet'):

    if backbone_type == 'alexnet':
        return torch.nn.Sequential(
            nn.Dropout(),
            nn.Linear(output_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, cls_num)
        )

    if backbone_type == 'vgg':
        return torch.nn.Sequential(
            nn.Dropout(),
            nn.Linear(output_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, cls_num)
        )

    if backbone_type == 'squeezenet':
        return torch.nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(output_size, 1000, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=1000, out_features=1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cls_num)
        )

    if backbone_type == 'mobilenet':
        return torch.nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=output_size, out_features=1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cls_num)
        )

    if backbone_type == 'resnet':
        return torch.nn.Sequential(
            nn.Linear(in_features=output_size, out_features=1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cls_num)
        )

    if backbone_type == 'shufflenet':
        return torch.nn.Sequential(
            nn.Linear(in_features=output_size, out_features=1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cls_num)
        )

    if backbone_type == 'inception':
        return torch.nn.Sequential(
            nn.Dropout(),
            nn.Linear(output_size, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cls_num)
        )

    if backbone_type == 'googlenet':
        return torch.nn.Sequential(
            nn.Dropout(),
            nn.Linear(output_size, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cls_num)
        )

    if backbone_type == 'densenet':
        return torch.nn.Sequential(
            nn.Linear(in_features=output_size, out_features=cls_num, bias=True)
        )

    print('Unsupported backbone.')
    exit()
