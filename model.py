import os
import torch
import torch.nn as nn
import torchvision.models as models
from datasets import load_dataset
# from classifier import Classifier
from utils import url_download, create_dir, model_dir


def get_backbone(ver, backbone_list):
    for bb in backbone_list:
        if ver == bb['ver']:
            return bb

    print('Backbone name not found, using default option - alexnet.')
    return backbone_list[0]


def model_info(backbone_ver):
    backbone_list = load_dataset(
        path="george-chou/vi_backbones",
        split="IMAGENET1K_V1"
    )
    backbone = get_backbone(backbone_ver, backbone_list)
    m_name = str(backbone['name'])
    m_type = str(backbone['type'])
    input_size = int(backbone['input_size'])
    m_url = str(backbone['url'])

    return m_name, m_type, input_size, m_url


def download_model(pre_model_url):
    pre_model_path = model_dir + '/' + (pre_model_url.split('/')[-1])
    create_dir(model_dir)

    if not os.path.exists(pre_model_path):
        url_download(pre_model_url, pre_model_path)

    return pre_model_path


def Classifier(cls_num, output_size):
    q = (1.0 * output_size / cls_num) ** 0.25
    l1 = int(q * cls_num)
    l2 = int(q * l1)
    l3 = int(q * l2)

    return torch.nn.Sequential(
        nn.Dropout(),
        nn.Linear(output_size, l3),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(l3, l2),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(l2, l1),
        nn.ReLU(inplace=True),
        nn.Linear(l1, cls_num)
    )


class Net():
    model = None
    m_type = ''
    m_name = ''
    m_url = ''
    input_size = 224
    output_size = 512
    training = True
    deep_finetune = False

    def __init__(self, cls_num, m_ver='alexnet', saved_model_path='', deep_finetune=False):
        self.training = (saved_model_path == '')
        self.deep_finetune = deep_finetune
        self.m_name, self.m_type, self.input_size, self.m_url = model_info(
            m_ver)

        if not hasattr(models, self.m_name):
            print('Unsupported model.')
            exit()

        self.model = eval('models.%s()' % m_ver)
        self._set_outsize()

        if self.training:
            pre_model_path = download_model(self.m_url)
            checkpoint = torch.load(pre_model_path, map_location='cpu')
            if torch.cuda.is_available():
                checkpoint = torch.load(pre_model_path)

            self.model.load_state_dict(checkpoint, False)

            for parma in self.model.parameters():
                parma.requires_grad = self.deep_finetune

            self._set_classifier(cls_num)
            self.model.train()

        else:
            self._set_classifier(cls_num)
            checkpoint = torch.load(saved_model_path, map_location='cpu')
            if torch.cuda.is_available():
                checkpoint = torch.load(saved_model_path)

            self.model.load_state_dict(checkpoint, False)
            self.model.eval()

    def _set_outsize(self):
        for name, module in self.model.named_modules():
            if str(name).__contains__('classifier') or str(name).__eq__('fc') or str(name).__contains__('head'):
                if isinstance(module, torch.nn.Linear):
                    self.output_size = module.in_features
                    print(
                        f"{name}(Linear): {self.output_size} -> {module.out_features}")
                    break

                if isinstance(module, torch.nn.Conv2d):
                    self.output_size = module.in_channels
                    print(
                        f"{name}(Conv2d): {self.output_size} -> {module.out_channels}")
                    break

    def _set_classifier(self, cls_num):
        if hasattr(self.model, 'classifier'):
            self.model.classifier = Classifier(cls_num, self.output_size)
            return

        elif hasattr(self.model, 'fc'):
            self.model.fc = Classifier(cls_num, self.output_size)
            return

        elif hasattr(self.model, 'head'):
            self.model.head = Classifier(cls_num, self.output_size)
            return

        self.model.heads.head = Classifier(cls_num, self.output_size)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
            self.model = self.model.cuda()

        if self.m_type == 'googlenet' and self.training:
            return self.model(x)[0]

        else:
            return self.model(x)

    def parameters(self):
        if self.deep_finetune:
            return self.model.parameters()

        if hasattr(self.model, 'classifier'):
            return self.model.classifier.parameters()

        if hasattr(self.model, 'fc'):
            return self.model.fc.parameters()

        print('Classifier part not found.')
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()
