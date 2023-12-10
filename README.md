# Piano-Classification
[![Python application](https://github.com/monet-joe/Piano-Classification/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monet-joe/Piano-Classification/actions/workflows/python-app.yml)
[![license](https://img.shields.io/github/license/monet-joe/Piano-Classification.svg)](https://github.com/monet-joe/Piano-Classification/blob/master/LICENSE)

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
git clone https://github.com/monet-joe/Piano-Classification.git
cd Piano-Classification
```

### Train
Assign a backbone(take squeezenet1_1 as an example) after `--model` to start training:
```
python train.py --model squeezenet1_1 --fullfinetune True
```
`--fullfinetune True` means full finetune, `False` means linear probing

<a href="https://huggingface.co/datasets/monet-joe/cv_backbones" target="_blank">Supported backbones</a>

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
A demo result of SqueezeNet fine-tuning:
|             Results              |                                                      Plots                                                       |
| :------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
|            Loss curve            | ![image](https://github.com/monet-joe/Piano-Classification/assets/20459298/8e80bb9e-60f9-40e0-a6a5-ad491f33074a) |
| Training and validation accuracy | ![image](https://github.com/monet-joe/Piano-Classification/assets/20459298/10dbfa66-cc8a-40be-a181-2e029a6064be) |
|         Confusion matrix         | ![image](https://github.com/monet-joe/Piano-Classification/assets/20459298/d925dc8d-952e-4919-8838-a6bc2e621f93) |

## Cite
```
@misc{zhou2023holistic,
      title={A Holistic Evaluation of Piano Sound Quality}, 
      author={Monan Zhou and Shangda Wu and Shaohua Ji and Zijin Li and Wei Li},
      year={2023},
      eprint={2310.04722},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
