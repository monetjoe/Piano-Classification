import os
import csv
import wave
import shutil
import random
import librosa
import contextlib
import librosa.display
import torch.utils.data
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from math import ceil
from utils import *


def load_cls():
    cls = []
    for _, dirnames, _ in os.walk(audio_dir):
        for dirname in dirnames:
            if len(os.listdir(audio_dir + '/' + dirname)) > 0:
                cls.append(dirname)

    return cls


classes = load_cls()


def calc_alpha(use_softmax=False):
    sample_sizes = []
    for _, dirnames, _ in os.walk(tra_dir):
        for dirname in dirnames:
            data_num = len(os.listdir(tra_dir + '/' + dirname))
            if data_num > 0:
                sample_sizes.append([data_num])

    if len(sample_sizes) <= 1:  # at least 2 classes are required
        print('Corrupt dataset.')
        exit()

    data_sizes = 1.0 / torch.tensor(sample_sizes)
    if use_softmax:
        data_sizes = F.softmax(data_sizes)

    else:
        data_sizes = data_sizes / data_sizes.sum()

    return data_sizes


def trans(audio_dir, img_dir, force_reload=True):
    print('Pre-processing data...')
    create_dir(img_dir)

    if len(os.listdir(img_dir)) > 0 and (not force_reload):
        print('Data already pre-processed.')
        return

    for _, dirnames, _ in os.walk(audio_dir):
        for dirname in dirnames:
            trans_files(audio_dir + '/' + dirname, img_dir)

    print('Data pre-processed.')


def save_audio_dur(audio_name, dur):
    dur_path = "./results/dur.csv"
    if not os.path.exists(dur_path):
        with open(dur_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([audio_name, dur])


def trans_files(cls_dir, img_dir):
    for _, _, filenames in os.walk(cls_dir):
        print('Converting ' + cls_dir.split('/')[-1] + '...')
        for filename in filenames:
            audio_name, dur = to_mel(
                cls_dir + '/' + filename, img_dir, width=0.2)
            if not os.path.exists('./results'):
                os.mkdir('./results')
            save_audio_dur(audio_name, dur)


def get_duration_wav(audio_path):
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return round(duration, 1)   # One decimal place


def to_mel(audio_path, img_dir, width=1.0, step=0.2):
    dur = get_duration_wav(audio_path)
    audio_name = audio_path.split('/')[-1][:-4]
    cls_name = audio_path.split('/')[-2]
    print('Duration of audio ' + audio_name + ': ' + str(dur) + 's')
    for i in np.arange(0.0, dur - width + step, step):
        index = round(i, 1)
        outpath = img_dir + '/' + cls_name + '__' + \
            audio_name + '[' + str(index) + '].png'
        y, sr = librosa.load(audio_path, offset=index, duration=width)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect)
        plt.axis('off')
        plt.savefig(outpath, bbox_inches='tight', pad_inches=-0.1)
        plt.close()

    return audio_name, dur


def embedding(file_path, batch_size=4, input_size=224):
    # dataset
    transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not len(os.listdir(file_path)) == len(classes):
        print('Corrupt ' + file_path.split('/')[-1] + 'set.')
        exit()

    inputSet = ImageFolder(file_path,  transform=transform)
    return torch.utils.data.DataLoader(inputSet, batch_size, shuffle=True, num_workers=2)


def load_data(img_dir, data_dir, force_reload=True):
    print('Loading data...')

    if os.path.exists(data_dir) and not force_reload:
        print('Data already loaded.')
        return

    if os.path.exists(data_dir) and force_reload:
        shutil.rmtree(data_dir)

    create_dir(data_dir)
    create_dir(tra_dir)
    create_dir(val_dir)
    create_dir(tes_dir)

    for _, _, filenames in os.walk(img_dir):

        length = len(filenames)
        if length < (10 * len(classes)):
            print('Insufficient data.')
            exit()

        p10 = ceil(length / 10)  # 10% of all data
        p20 = 2 * p10            # 20% of all data

        val_test = random.sample(filenames, p20)
        trainset = list(set(filenames) - set(val_test))

        print('Copying validation data...')
        for i in range(p10):
            copy_img(val_test[i], val_dir)

        print('Copying test data...')
        for i in range(p10, p20):
            copy_img(val_test[i], tes_dir)

        print('Copying training data...')
        for filename in trainset:
            copy_img(filename, tra_dir)

    print('Data loaded.')


def copy_img(img_name, tag_dir):
    # cls_id = int(img_name[0])
    # cls = str(cls_id) + '_' + classes[cls_id - 1]
    cls = img_name.split('__')[0]
    outdir = tag_dir + '/' + cls
    create_dir(outdir)
    shutil.copy('./image/' + img_name, outdir)


def prepare_data(batch_size=4, input_size=224):

    if (not os.path.exists(audio_dir)):
        unzip_file('./audio.zip', './')

    trans(audio_dir, img_dir, force_reload=False)
    load_data(img_dir, data_dir, force_reload=False)

    print('Embedding data...')
    trainLoader = embedding(tra_dir, batch_size, input_size)
    validLoader = embedding(val_dir, batch_size, input_size)
    testLoader = embedding(tes_dir, batch_size, input_size)
    print('Data embedded.')

    return trainLoader, validLoader, testLoader


if __name__ == "__main__":
    trans(audio_dir, img_dir, force_reload=False)
    load_data(img_dir, data_dir)
