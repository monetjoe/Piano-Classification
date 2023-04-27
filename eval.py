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
from utils import time_stamp, create_dir, results_dir
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


def split_embed(audio_path, input_size, width=1.0, step=0.2):
    cache_dir = audio_path[:-4] + '_' + time_stamp()
    dur = get_duration_wav(audio_path)
    start = 0  # (dur - width) * 0.5
    end = dur - width + step  # (dur + width) * 0.5
    inputs = []
    create_dir(cache_dir)
    for i in np.arange(start, end, step):
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


def get_saved_model(log_dir, history):
    if valid_path(log_dir, history):
        m_ver = history.split('__')[0]
        history = '/' + history
    else:
        history, m_ver = get_latest_log(log_dir)

    saved_model_path = log_dir + history + '/save.pt'

    if not os.path.exists(saved_model_path):
        print('No history found, start a new term of training...')
        train()

    return saved_model_path, m_ver


def eval(tag='', log_dir=results_dir, history='', split_mode=False, cls_num=len(classes)):

    if not os.path.exists(tag):
        print('Target not found.')
        exit()

    saved_model_path, m_ver = get_saved_model(log_dir, history)
    model = Net(m_ver, saved_model_path, cls_num=cls_num)
    print('[' + m_ver + '] prediction result:')

    if split_mode:
        inputs = split_embed(tag, model.input_size, width=0.2)
        outputs = []
        preds = torch.zeros(cls_num)

        if torch.cuda.is_available():
            preds = preds.cuda()

        for input in inputs:
            output = model.forward(input)
            preds += (output.data)[0]
            pred_id = torch.max(output.data, 1)[1]
            prediction = classes[pred_id]
            outputs.append(prediction)

        print(max(outputs, key=outputs.count))
        return preds / len(inputs)

    else:
        input = embed(tag, model.input_size)
        output = model.forward(input)
        pred_id = torch.max(output.data, 1)[1]
        prediction = classes[pred_id]
        print(prediction)
        return output[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--target', type=str, default='./test/KAWAI.wav')
    parser.add_argument('--log', type=str, default='')
    args = parser.parse_args()

    eval(tag=args.target, history=args.log, split_mode=True)
