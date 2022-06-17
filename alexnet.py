import os
import torch
import requests
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from data import create_dir


def url_download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_model():
    pre_model_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    model_dir = './model/'
    pre_model_path = model_dir + 'alexnet-owt-4df8aa71.pth'

    create_dir(model_dir)

    if not os.path.exists(pre_model_path):
        url_download(pre_model_url, pre_model_path)

    return pre_model_path


def AlexNet():

    pre_model_path = download_model()

    model = models.alexnet(pretrained=True)
    model.load_state_dict(torch.load(pre_model_path))

    for parma in model.parameters():
        parma.requires_grad = False

    model.classifier = torch.nn.Sequential(nn.Dropout(),
                                           nn.Linear(256 * 6 * 6, 4096),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(),
                                           nn.Linear(4096, 4096),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(0.5),
                                           nn.Linear(4096, 1000),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(1000, 7))

    return model
