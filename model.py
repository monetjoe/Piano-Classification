import os
import torch
import torch.nn as nn
import torchvision.models as models
from modelscope.msdatasets import MsDataset
from utils import url_download, create_dir, MODEL_DIR


class WCE(nn.CrossEntropyLoss):
    def __init__(self, sample_sizes: list):
        super(WCE, self).__init__()
        weights = torch.tensor(
            [1.0 / size for size in sample_sizes], dtype=torch.float32
        )
        self.weight = weights / weights.sum()


class Net:
    model: nn.Module = None
    m_type = "squeezenet1_1"
    m_url = "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth"
    input_size = 224
    output_size = 512
    training = True
    full_finetune = False

    def __init__(
        self,
        cls_num,
        m_ver="squeezenet1_1",
        saved_model_path="",
        full_finetune=False,
    ):
        self.training = saved_model_path == ""
        self.full_finetune = full_finetune
        self.m_type, self.input_size, self.m_url = self._model_info(m_ver)

        if not hasattr(models, m_ver):
            print("Unsupported model.")
            exit()

        self.model = eval("models.%s()" % m_ver)
        linear_output = self._set_outsize()

        if self.training:
            pre_model_path = self._download_model(self.m_url)
            if torch.cuda.is_available():
                checkpoint = torch.load(pre_model_path)
            else:
                checkpoint = torch.load(pre_model_path, map_location="cpu")

            self.model.load_state_dict(checkpoint, False)

            for parma in self.model.parameters():
                parma.requires_grad = self.full_finetune

            self._set__classifier(cls_num, linear_output)
            self.model.train()

        else:
            self._set__classifier(cls_num, linear_output)
            checkpoint = torch.load(saved_model_path, map_location="cpu")
            if torch.cuda.is_available():
                checkpoint = torch.load(saved_model_path)

            self.model.load_state_dict(checkpoint, False)
            self.model.eval()

    def _get_backbone(self, ver, backbone_list):
        for bb in backbone_list:
            if ver == bb["ver"]:
                return bb

        print("Backbone name not found, using default option - alexnet.")
        return backbone_list[0]

    def _model_info(self, backbone_ver):
        backbone_list = MsDataset.load(
            "monetjoe/cv_backbones",
            subset_name="default",
            split="v1",
        )
        backbone = self._get_backbone(backbone_ver, backbone_list)
        m_type = str(backbone["type"])
        input_size = int(backbone["input_size"])
        m_url = str(backbone["url"])

        return m_type, input_size, m_url

    def _download_model(self, pre_model_url):
        pre_model_path = f"{MODEL_DIR }/{pre_model_url.split('/')[-1]}"
        create_dir(MODEL_DIR)

        if not os.path.exists(pre_model_path):
            url_download(pre_model_url, pre_model_path)

        return pre_model_path

    def _classifier(self, cls_num: int, output_size: int, linear_output: bool):
        q = (1.0 * output_size / cls_num) ** 0.25
        l1 = int(q * cls_num)
        l2 = int(q * l1)
        l3 = int(q * l2)

        if linear_output:
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
                nn.Linear(l1, cls_num),
            )

        else:
            return torch.nn.Sequential(
                nn.Dropout(),
                nn.Conv2d(output_size, l3, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(l3, l2),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(l2, l1),
                nn.ReLU(inplace=True),
                nn.Linear(l1, cls_num),
            )

    def _set_outsize(self, debug_mode=False):
        for name, module in self.model.named_modules():
            if (
                str(name).__contains__("classifier")
                or str(name).__eq__("fc")
                or str(name).__contains__("head")
            ):
                if isinstance(module, torch.nn.Linear):
                    self.output_size = module.in_features
                    if debug_mode:
                        print(
                            f"{name}(Linear): {self.output_size} -> {module.out_features}"
                        )
                    return True

                if isinstance(module, torch.nn.Conv2d):
                    self.output_size = module.in_channels
                    if debug_mode:
                        print(
                            f"{name}(Conv2d): {self.output_size} -> {module.out_channels}"
                        )
                    return False

        return False

    def _set__classifier(self, cls_num, linear_output):
        if hasattr(self.model, "classifier"):
            self.model.classifier = self._classifier(
                cls_num, self.output_size, linear_output
            )
            return

        elif hasattr(self.model, "fc"):
            self.model.fc = self._classifier(cls_num, self.output_size, linear_output)
            return

        elif hasattr(self.model, "head"):
            self.model.head = self._classifier(cls_num, self.output_size, linear_output)
            return

        self.model.heads.head = self._classifier(
            cls_num, self.output_size, linear_output
        )

    def _get_insize(self):
        return self.input_size

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
            self.model = self.model.cuda()

        if self.m_type == "googlenet" and self.training:
            return self.model(x)[0]
        else:
            return self.model(x)

    def parameters(self):
        if self.full_finetune:
            return self.model.parameters()

        if hasattr(self.model, "classifier"):
            return self.model.classifier.parameters()

        if hasattr(self.model, "fc"):
            return self.model.fc.parameters()

        print("Classifier part not found.")
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()
