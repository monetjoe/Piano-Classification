import os
import torch
from datasets import load_dataset
from classifier import Classifier
from utils import url_download, create_dir, model_dir
# from data import classes
# save below line, it is called by class Net() hiddenly
import torchvision.models as models


def get_backbone(ver, backbone_list):
    for bb in backbone_list:
        if ver == bb['ver']:
            return bb

    print('Backbone name not found, using default option - alexnet.')
    return backbone_list[0]


def model_info(backbone_ver):
    backbone_list = load_dataset("george-chou/CNN_backbones", split="train")
    backbone = get_backbone(backbone_ver, backbone_list)
    input_size = int(backbone['input_size'])
    output_size = int(backbone['output_size'])
    m_type = str(backbone['type'])
    m_url = str(backbone['url'])

    return m_url, m_type, input_size, output_size


def download_model(pre_model_url):
    pre_model_path = model_dir + '/' + (pre_model_url.split('/')[-1])
    create_dir(model_dir)

    if not os.path.exists(pre_model_path):
        url_download(pre_model_url, pre_model_path)

    return pre_model_path


def set_classifier(model, output_size, m_type, cls_num):
    if hasattr(model, 'classifier'):
        model.classifier = Classifier(
            cls_num, output_size, backbone_type=m_type)

    elif hasattr(model, 'fc'):
        model.fc = Classifier(cls_num, output_size, backbone_type=m_type)


class Net():
    model = None
    m_url, m_type = '', ''
    input_size = 224
    output_size = 0
    training = True
    deep_finetune = False

    def __init__(self, cls_num, m_ver='alexnet', saved_model_path='', deep_finetune=False):

        self.training = (saved_model_path == '')
        self.deep_finetune = deep_finetune
        self.m_url, self.m_type, self.input_size, self.output_size = model_info(
            m_ver)
        self.model = eval('models.%s()' % m_ver)

        if self.training:
            pre_model_path = download_model(self.m_url)
            checkpoint = torch.load(pre_model_path, map_location='cpu')
            if torch.cuda.is_available():
                checkpoint = torch.load(pre_model_path)

            self.model.load_state_dict(checkpoint, False)

            for parma in self.model.parameters():
                parma.requires_grad = self.deep_finetune

            set_classifier(self.model, self.output_size, self.m_type, cls_num)
            self.model.train()

        else:
            set_classifier(self.model, self.output_size, self.m_type, cls_num)
            checkpoint = torch.load(saved_model_path, map_location='cpu')
            if torch.cuda.is_available():
                checkpoint = torch.load(saved_model_path)

            self.model.load_state_dict(checkpoint, False)
            self.model.eval()

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
