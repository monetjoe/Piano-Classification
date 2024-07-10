# Piano-Classification
[![Python application](https://github.com/monetjoe/Piano-Classification/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monetjoe/Piano-Classification/actions/workflows/python-app.yml)
[![license](https://img.shields.io/github/license/monetjoe/Piano-Classification.svg)](https://github.com/monetjoe/Piano-Classification/blob/master/LICENSE)
[![](https://img.shields.io/badge/HF-pianos-ffd21e.svg)](https://huggingface.co/spaces/ccmusic-database/pianos)
[![](https://img.shields.io/badge/ModelScope-pianos-624aff.svg)](https://www.modelscope.cn/studios/ccmusic-database/pianos)
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
git clone git@github.com:monetjoe/Piano-Classification.git
cd Piano-Classification
```

### Train
Assign a backbone(take squeezenet1_1 as an example) after `--model` to start training:
```bash
python train.py --model squeezenet1_1 --fullfinetune True
```
`--fullfinetune True` means full finetune, `False` means linear probing

#### Supported backbones
<a href="https://huggingface.co/datasets/monetjoe/cv_backbones" target="_blank">Mirror 1</a><br>
<a href="https://www.modelscope.cn/datasets/monetjoe/cv_backbones/dataPeview" target="_blank">Mirror 2</a>

### Plot results
After finishing the training, use the below command to plot the latest results:
```bash
python plot.py
```

## Results
A demo result of SqueezeNet fine-tuning:
|             Results              |                                                            Plots                                                             |
| :------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: |
|            Loss curve            | <img src="https://www.modelscope.cn/api/v1/models/ccmusic-database/pianos/repo?Revision=master&FilePath=loss.jpg&View=true"> |
| Training and validation accuracy | <img src="https://www.modelscope.cn/api/v1/models/ccmusic-database/pianos/repo?Revision=master&FilePath=acc.jpg&View=true">  |
|         Confusion matrix         | <img src="https://www.modelscope.cn/api/v1/models/ccmusic-database/pianos/repo?Revision=master&FilePath=mat.jpg&View=true">  |

## Cite
```bibtex
@inproceedings{DBLP:journals/corr/abs-2310-04722,
  author    = {Monan Zhou and
               Shangda Wu and
               Shaohua Ji and
               Zijin Li and
               Wei Li},
  title     = {A Holistic Evaluation of Piano Sound Quality},
  booktitle = {Proceedings of the 10th Conference on Sound and Music Technology (CSMT)},
  year      = {2023},
  publisher = {Springer Singapore},
  address   = {Singapore},
  timestamp = {Fri, 20 Oct 2023 12:04:38 +0200},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
