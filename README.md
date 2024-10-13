# Piano-Classification
[![license](https://img.shields.io/github/license/monetjoe/Piano-Classification.svg)](https://github.com/monetjoe/Piano-Classification/blob/master/LICENSE)
[![Python application](https://github.com/monetjoe/Piano-Classification/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monetjoe/Piano-Classification/actions/workflows/python-app.yml)
[![](https://img.shields.io/badge/HuggingFace-pianos-ffd21e.svg)](https://huggingface.co/spaces/ccmusic-database/pianos)
[![](https://img.shields.io/badge/ModelScope-pianos-624aff.svg)](https://www.modelscope.cn/studios/ccmusic-database/pianos)
[![](https://img.shields.io/badge/arxiv-2310.04722-b31b1b.svg)](https://arxiv.org/pdf/2310.04722.pdf)

Classify piano sound quality by fine-tuned pre-trained CNN models.

## Requirements
```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install -r requirements.txt
```

## Usage
### Maintenance
```bash
git clone git@github.com:monetjoe/Piano-Classification.git
cd Piano-Classification
```

### Train
Assign a backbone(take squeezenet1_1 as an example) after `--model` to start training:
```bash
python train.py --model squeezenet1_1 --fullfinetune True --fl True
```
`--fullfinetune True` means full finetune, `False` means linear probing<br>
`--fl True` means using focal loss

#### Supported backbones
| <a href="https://huggingface.co/datasets/monetjoe/cv_backbones" target="_blank">Mirror 1</a> | <a href="https://www.modelscope.cn/datasets/monetjoe/cv_backbones/dataPeview" target="_blank">Mirror 2</a> |
| :------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: |

### Plot results
After finishing the training, use the below command to plot the latest results:
```bash
python plot.py
```

## Results
A demo result of SqueezeNet fine-tuning:
|             Results              |                                        Plots                                         |
| :------------------------------: | :----------------------------------------------------------------------------------: |
|            Loss curve            | ![](https://github.com/user-attachments/assets/f6893fdd-9315-44c7-850f-6a29ebdc8c15) |
| Training and validation accuracy | ![](https://github.com/user-attachments/assets/07c7fb83-156c-40f8-9372-f96a818eeb39) |
|         Confusion matrix         | ![](https://github.com/user-attachments/assets/284b82e5-bb45-44f1-8bdc-7d2832d3e6c3) |

## Cite
```bibtex
@inproceedings{Zhou2023AHE,
  author    = {Monan Zhou and Shangda Wu and Shaohua Ji and Zijin Li and Wei Li},
  title     = {A Holistic Evaluation of Piano Sound Quality},
  booktitle = {Proceedings of the 10th Conference on Sound and Music Technology (CSMT)},
  year      = {2023},
  publisher = {Springer Singapore},
  address   = {Singapore}
}
```
