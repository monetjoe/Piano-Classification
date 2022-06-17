import os
import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
from train import train, time_stamp
from plotter import get_latest_log, valid_path
from data import get_duration_wav, classes
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

parser = argparse.ArgumentParser(description='predict')
parser.add_argument('--target', type=str,
                    default='./test/KAWAI-C4.wav', help='Select wav to be predicted.')
parser.add_argument('--log', type=str,
                    default='', help='Select a training history.')
args = parser.parse_args()


def embed(audio_path):

    img_path = audio_path[:-4] + '_' + time_stamp() + '.png'
    dur = get_duration_wav(audio_path)
    y, sr = librosa.load(audio_path, offset=0.0, duration=dur)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect)
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=-0.1)
    plt.close()

    return embed_img(img_path)


def embed_img(img_path):
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.RandomAffine(5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(img_path).convert("RGB")

    if os.path.exists(img_path):
        os.remove(img_path)

    return transform(img)


def eval(log_dir='./logs', latest_log=''):

    if valid_path(log_dir, latest_log):
        latest_log = '/' + latest_log
    else:
        latest_log = get_latest_log(log_dir)

    saved_model_path = log_dir + latest_log + '/save.pt'

    if not os.path.exists(saved_model_path):
        print('No history found, start a new term of training...')
        train()

    model = torch.load(saved_model_path)
    torch.cuda.empty_cache()
    model = model.cuda()
    input = embed(args.target).cuda()
    output = model(input.unsqueeze(0))
    predict = torch.max(output.data, 1)[1]
    print(classes[predict])


if __name__ == "__main__":
    # eval(latest_log=args.log)
    eval()
