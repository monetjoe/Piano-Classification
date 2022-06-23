from train import train, time_stamp
import os
import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
from plotter import get_latest_log, valid_path
from data import create_dir, get_duration_wav, classes, input_size
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from model import Net_eval, backbone_network


def embed(audio_path):
    img_path = audio_path[:-4] + '_' + time_stamp() + '.png'
    dur = get_duration_wav(audio_path)
    y, sr = librosa.load(audio_path, offset=0.0, duration=dur)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect)
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=-0.1)
    plt.close()

    return embed_img(img_path)


def split_embed(audio_path, width=0.2, step=0.2):
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
        inputs.append(embed_img(outpath).unsqueeze(0))

    os.rmdir(cache_dir)
    return inputs


def embed_img(img_path, rm_cache=True):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(300),
        # transforms.RandomAffine(5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(img_path).convert("RGB")

    if rm_cache and os.path.exists(img_path):
        os.remove(img_path)

    return transform(img)


def eval(log_dir='./logs', history='', split_mode=False):

    tag = args.target

    if not os.path.exists(tag):
        print('Target not found.')
        exit()

    if valid_path(log_dir, history):
        history = '/' + history
    else:
        history = get_latest_log(log_dir)

    saved_model_path = log_dir + history + '/save.pt'

    if not os.path.exists(saved_model_path):
        print('No history found, start a new term of training...')
        train()

    model = Net_eval(saved_model_path, backbone_network)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.cuda()

    if not split_mode:
        input = embed(tag).unsqueeze(0)

        if torch.cuda.is_available():
            input = input.cuda()

        output = model(input)
        pred_id = torch.max(output.data, 1)[1]
        predict = classes[pred_id]
        print(predict)

    else:
        inputs = split_embed(tag)
        outputs = []
        for input in inputs:

            if torch.cuda.is_available():
                input = input.cuda()

            output = model(input)
            pred_id = torch.max(output.data, 1)[1]
            predict = classes[pred_id]
            outputs.append(predict)

        # print(outputs)
        print(max(outputs, key=outputs.count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--target', type=str,
                        default='./test/KAWAI.wav', help='Select wav to be predicted.')
    parser.add_argument('--log', type=str,
                        default='', help='Select a training history.')
    args = parser.parse_args()
    eval(history=args.log, split_mode=True)
    # eval()
