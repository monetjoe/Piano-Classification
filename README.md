# Piano-Classification
[![Python application](https://github.com/monet-joe/Piano-Classification/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monet-joe/Piano-Classification/actions/workflows/python-app.yml)
[![license](https://img.shields.io/github/license/monet-joe/Piano-Classification.svg)](https://github.com/monet-joe/Piano-Classification/blob/master/LICENSE)
[![](https://img.shields.io/badge/HF-pianos-ffd21e.svg)](https://huggingface.co/spaces/ccmusic-database/pianos)
[![](https://img.shields.io/badge/ModelScope-pianos-624aff.svg)](https://www.modelscope.cn/studios/ccmusic/pianos)
[![](https://img.shields.io/badge/arxiv-2310.04722-b31b1b.svg)](https://arxiv.org/pdf/2310.04722.pdf)

Classify piano sound quality by fine-tuned pre-trained CNN models.

## Requirements
```bash
conda create -n cnn --yes --file conda.txt
conda activate cnn
pip install -r requirements.txt
```

## Usage
### Maintenance
```bash
git clone git@gitee.com:MuGeminorum/Piano-Classification.git
cd Piano-Classification
```

### Train
Assign a backbone(take squeezenet1_1 as an example) after `--model` to start training:
```bash
python train.py --model squeezenet1_1 --fullfinetune True
```
`--fullfinetune True` means full finetune, `False` means linear probing

<a href="https://www.modelscope.cn/datasets/monetjoe/cv_backbones/dataPeview" target="_blank">Supported backbones</a>

### Plot results
After finishing the training, use below command to plot latest results:
```bash
python plot.py
```

## Results
A demo result of SqueezeNet fine-tuning:
|             Results              |                                                      Plots                                                       |
| :------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
|            Loss curve            | ![image](https://github.com/monet-joe/Piano-Classification/assets/20459298/8e80bb9e-60f9-40e0-a6a5-ad491f33074a) |
| Training and validation accuracy | ![image](https://github.com/monet-joe/Piano-Classification/assets/20459298/10dbfa66-cc8a-40be-a181-2e029a6064be) |
|         Confusion matrix         | ![image](https://github.com/monet-joe/Piano-Classification/assets/20459298/d925dc8d-952e-4919-8838-a6bc2e621f93) |

## Cite
```bash
@misc{zhou2023holistic,
      title={A Holistic Evaluation of Piano Sound Quality}, 
      author={Monan Zhou and Shangda Wu and Shaohua Ji and Zijin Li and Wei Li},
      year={2023},
      eprint={2310.04722},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
