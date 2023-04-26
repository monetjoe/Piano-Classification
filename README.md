# Piano-Classification

[![Python application](https://github.com/george-chou/Piano-Classification/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/george-chou/Piano-Classification/actions/workflows/python-app.yml)
[![license](https://img.shields.io/github/license/george-chou/Piano-Classification.svg)](https://github.com/george-chou/Piano-Classification/blob/master/LICENSE)

Classify piano sound quality by fine-tuned pre-trained CNN models.

## Requirements
```
echo y | conda create -n piano-cls python=3.9
conda activate piano-cls
echo y | conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Usage

### Code download

```
git clone https://github.com/george-chou/Piano-Classification.git
cd Piano-Classification
```
### Dataset download

Download [dataset](https://github.com/george-chou/Piano-Classification/releases/download/dataset/audio.zip) and extract it into the project path as following directory structure:

- Piano-Classification
    - audio
        - 1_ZhuJiang
          - 1009.wav
          - 1010.wav
          - ...     
        - 2_Old-YingChang
          - 2009.wav
          - 2010.wav
          - ...       
        - 3_Steinway-Theater
        - 4_StarSea
        - 5_KAWAI
        - 6_Steinway
        - 7_KAWAI-Tri
        - 8_Yamaha

### Train
Assign a backbone(for example inception_v3) after `--model` to start training:
```
python train.py --model inception_v3
```

__Supported backbones__
| Ver                | Type       |
| :----------------- | :--------- |
| alexnet            | AlexNet    |
| vgg11              | VGG        |
| vgg13              | VGG        |
| vgg16              | VGG        |
| vgg19              | VGG        |
| vgg11_bn           | VGG        |
| vgg13_bn           | VGG        |
| vgg16_bn           | VGG        |
| vgg19_bn           | VGG        |
| resnet18           | ResNet     |
| resnet34           | ResNet     |
| resnet50           | ResNet     |
| resnet101          | ResNet     |
| resnet152          | ResNet     |
| resnext50_32x4d    | ResNet     |
| resnext101_32x8d   | ResNet     |
| wide_resnet50_2    | ResNet     |
| wide_resnet101_2   | ResNet     |
| squeezenet1_0      | SqueezeNet |
| squeezenet1_1      | SqueezeNet |
| densenet121        | DenseNet   |
| densenet169        | DenseNet   |
| densenet201        | DenseNet   |
| densenet161        | DenseNet   |
| googlenet          | GoogleNet  |
| inception_v3       | GoogleNet  |
| shufflenet_v2_x0_5 | ShuffleNet |
| shufflenet_v2_x1_0 | ShuffleNet |
| mobilenet_v2       | MobileNet  |
| mnasnet0_5         | MobileNet  |
| mnasnet1_0         | MobileNet  |

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
