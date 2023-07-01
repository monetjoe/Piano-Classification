# Piano-Classification

[![Python application](https://github.com/george-chou/Piano-Classification/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/george-chou/Piano-Classification/actions/workflows/python-app.yml)
[![license](https://img.shields.io/github/license/george-chou/Piano-Classification.svg)](https://github.com/george-chou/Piano-Classification/blob/master/LICENSE)

Classify piano sound quality by fine-tuned pre-trained CNN models.

## Requirements
```
echo y | conda create -n pianocls python=3.9
conda activate pianocls
echo y | conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3
pip install -r requirements.txt
```

## Usage

### Code download

```
git clone https://github.com/george-chou/Piano-Classification.git
cd Piano-Classification
```

### Train
Assign a backbone(take inception_v3 as an example) after `--model` to start training:
```
python train.py --model inception_v3
```

<a href="https://huggingface.co/datasets/george-chou/vi_backbones" target="_blank">Supported backbones</a>

### Plot results
After finishing the training, use below command to plot latest results:
```
python plot.py
```

### Predict
Use below command to predict an audio target by latest saved model:
```
python eval.py --target ./test/KAWAI.wav
```

## Results
A demo result of AlexNet fine-tuning:
|              Index               |                                                      Plot                                                      |
| :------------------------------: | :------------------------------------------------------------------------------------------------------------: |
|            Loss curve            | ![loss](https://user-images.githubusercontent.com/20459298/233117067-380e9921-3b6d-4542-a4a7-0ba92bb95534.png) |
| Training and validation accuracy | ![acc](https://user-images.githubusercontent.com/20459298/233117103-231c8555-1b95-49e1-938c-88eb5494d542.png)  |
|         Confusion matrix         | ![mat](https://user-images.githubusercontent.com/20459298/233117128-d6033719-a104-4830-95c1-0038cf0cc954.png)  |

## Cite
```
@article{CSMT2023HEPSQ,
  title={A Holistic Evaluation of Piano Sound Quality},
  author={Monan Zhou, Shangda Wu, Shaohua Ji, Zijin Li, Wei Li},
  year={2023}
}
```