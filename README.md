# Piano-Classification

[![Python application](https://github.com/george-chou/Piano-Classification/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/george-chou/Piano-Classification/actions/workflows/python-app.yml)
[![license](https://img.shields.io/github/license/george-chou/Piano-Classification.svg)](https://github.com/george-chou/Piano-Classification/blob/master/LICENSE)

Classify piano sound quality by fine-tuned pre-trained CNN models.

## Requirements
```
conda create -n cnn --yes --file conda.txt
conda activate cnn
pip install -r requirements.txt
```

## Usage

### Code download

```
git clone https://github.com/george-chou/Piano-Classification.git
cd Piano-Classification
```

### Train
Assign a backbone(take squeezenet1_1 as an example) after `--model` to start training:
```
python train.py --model squeezenet1_1
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
|            Loss curve            | [loss.pdf](https://github.com/george-chou/Piano-Classification/files/12640187/loss.pdf)
 |
| Training and validation accuracy | [acc.pdf](https://github.com/george-chou/Piano-Classification/files/12640186/acc.pdf)
  |
|         Confusion matrix         | [mat.pdf](https://github.com/george-chou/Piano-Classification/files/12640188/mat.pdf)
 |

## Cite
```
@article{CSMT2023HEPSQ,
  title={A Holistic Evaluation of Piano Sound Quality},[acc.pdf](https://github.com/george-chou/Piano-Classification/files/12640185/acc.pdf)

  author={Monan Zhou, Shangda Wu, Shaohua Ji, Zijin Li, Wei Li},
  year={2023}
}
```
