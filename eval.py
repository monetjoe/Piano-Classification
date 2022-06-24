import os
import torch
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from model import Net
from train import train
from utils import time_stamp, create_dir
from plot import get_latest_log, valid_path
from data import get_duration_wav, classes


def embed(audio_path, input_size):
    img_path = audio_path[:-4] + '_' + time_stamp() + '.png'
    dur = get_duration_wav(audio_path)
    y, sr = librosa.load(audio_path, offset=0.0, duration=dur)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect)
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=-0.1)
    plt.close()

    return embed_img(img_path, input_size)


def split_embed(audio_path, input_size, width=0.2, step=0.2):
    cache_dir = audio_path[:-4] + '_' + time_stamp()
    dur = get_duration_wav(audio_path)
    # start = (dur - width) * 0.5
    # end = (dur + width) * 0.5
    inputs = []
    create_dir(cache_dir)
    for i in np.arange(0, dur - width + step, step):
        index = round(i, 1)
        outpath = cache_dir + '/' + str(index) + '.png'
        y, sr = librosa.load(audio_path, offset=index, duration=width)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect)
        plt.axis('off')
        plt.savefig(outpath, bbox_inches='tight', pad_inches=-0.1)
        plt.close()
        inputs.append(embed_img(outpath, input_size))

    os.rmdir(cache_dir)
    return inputs


def embed_img(img_path, input_size=224, rm_cache=True):
    transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(img_path).convert("RGB")

    if rm_cache and os.path.exists(img_path):
        os.remove(img_path)

    return transform(img).unsqueeze(0)


def eval(log_dir='./logs', history='', split_mode=False):

    tag = args.target

    if not os.path.exists(tag):
        print('Target not found.')
        exit()

    if valid_path(log_dir, history):
        m_ver = history.split('__')[0]
        history = '/' + history
    else:
        history, m_ver = get_latest_log(log_dir)

    saved_model_path = log_dir + history + '/save.pt'

    if not os.path.exists(saved_model_path):
        print('No history found, start a new term of training...')
        train()

    model = Net(m_ver, saved_model_path)
    print('[' + m_ver + '] prediction result:')

    if split_mode:
        inputs = split_embed(tag, model.input_size)
        outputs = []
        for input in inputs:
            output = model.forward(input)
            pred_id = torch.max(output.data, 1)[1]
            prediction = classes[pred_id]
            outputs.append(prediction)

        print(max(outputs, key=outputs.count))

    else:
        input = embed(tag, model.input_size)
        output = model.forward(input)
        pred_id = torch.max(output.data, 1)[1]
        prediction = classes[pred_id]
        # print(output)
        print(prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--target', type=str, default='./test/KAWAI.wav')
    parser.add_argument('--log', type=str, default='')
    args = parser.parse_args()

    eval(history=args.log, split_mode=True)
