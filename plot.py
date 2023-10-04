import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as ss
from datasets import load_dataset
from utils import *

plt.rcParams['font.sans-serif'] = 'Times New Roman'


def show_point(max_id, list):
    show_max = '(' + str(max_id + 1) + ', ' + str(round(list[max_id], 2)) + ')'
    plt.annotate(show_max, xytext=(
        max_id + 1, list[max_id]), xy=(max_id + 1, list[max_id]), fontsize=6)


def smooth(y):
    if 95 <= len(y):
        return ss.savgol_filter(y, 95, 3)

    return y


def plot_acc(tra_acc_list, val_acc_list):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    max1 = np.argmax(y1)
    max2 = np.argmax(y2)

    plt.title('Accuracy of training and validation', fontweight='bold')
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")
    plt.plot(1+max1, y1[max1], 'r-o')
    plt.plot(1+max2, y2[max2], 'r-o')
    show_point(max1, y1)
    show_point(max2, y2)
    plt.legend()


def save_acc(tra_acc_list, val_acc_list, save_path):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    max1 = np.argmax(y1)
    max2 = np.argmax(y2)

    plt.title('Accuracy of training and validation', fontweight='bold')
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")
    plt.plot(1+max1, y1[max1], 'r-o')
    plt.plot(1+max2, y2[max2], 'r-o')
    show_point(max1, y1)
    show_point(max2, y2)
    plt.legend()
    plt.savefig(save_path + "/acc.pdf", bbox_inches='tight')
    plt.close()


def save_loss(loss_list, save_path):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.title('Loss curve', fontweight='bold')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(x_loss, smooth(loss_list))
    plt.savefig(save_path + "/loss.pdf", bbox_inches='tight')
    plt.close()


def plot_loss(loss_list):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.title('Loss curve', fontweight='bold')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(x_loss, smooth(loss_list))


def get_latest_log(path):
    if not os.path.exists(path):
        print('Please train a model first.')
        exit()

    lists = os.listdir(path)
    if len(lists) == 0:
        print('No history found in logs.')
        exit()

    lists = os.listdir(path)
    lists.sort(key=lambda x: os.path.getmtime((path + "\\" + x)))
    history = lists[-1]
    m_ver = history.split('__')[0]

    return '/' + history, m_ver


def valid_path(log_path, latest_log):
    if latest_log == '':
        return False

    dir_path = log_path + '/' + latest_log
    return os.path.exists(dir_path)


def load_history(log_dir=results_dir, latest_log=''):

    create_dir(log_dir)

    if len(os.listdir(log_dir)) == 0:
        print('Please finish training first.')
        exit()

    if valid_path(log_dir, latest_log):
        latest_log = '/' + latest_log
    else:
        latest_log, _ = get_latest_log(log_dir)

    latest_acc = log_dir + latest_log + '/acc.csv'
    latest_loss = log_dir + latest_log + '/loss.csv'
    acc_list = pd.read_csv(latest_acc)
    tra_acc_list = acc_list['tra_acc_list'].tolist()
    val_acc_list = acc_list['val_acc_list'].tolist()
    loss_list = pd.read_csv(latest_loss)['loss_list'].tolist()

    cm = np.loadtxt(open(log_dir + latest_log + "/mat.csv", "rb"),
                    delimiter=",", skiprows=0)

    return tra_acc_list, val_acc_list, loss_list, cm


def plot_confusion_matrix(cm, labels_name, title='Confusion matrix'):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # Normalized
    # Display an image on a specific window
    plt.imshow(cm, interpolation='nearest')
    plt.title(title, fontweight='bold')    # image caption
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    # print the labels on the x-axis coordinates
    plt.xticks(num_local, labels_name, rotation=90)
    # print the label on the y-axis coordinate
    plt.yticks(num_local, labels_name)
    plt.ylabel('true label')
    plt.xlabel('predicted label')


def save_confusion_matrix(cm, labels_name, save_path, title='Confusion matrix'):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # Normalized
    # Display an image on a specific window
    plt.imshow(cm, interpolation='nearest')
    plt.title(title, fontweight='bold')    # image caption
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    # print the labels on the x-axis coordinates
    plt.xticks(num_local, labels_name, rotation=90)
    # print the label on the y-axis coordinate
    plt.yticks(num_local, labels_name)
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.tight_layout()
    plt.savefig(save_path + '/mat.pdf', bbox_inches='tight')
    plt.close()


def plot_all(labels_name, latest_log=''):
    tra_acc_list, val_acc_list, loss_list, cm = load_history(
        latest_log=latest_log)

    plt.figure(figsize=(9, 7))
    plt.subplot(221)
    plot_acc(tra_acc_list, val_acc_list)
    plt.subplot(222)
    plot_loss(loss_list)
    plt.subplot(224)
    plot_confusion_matrix(cm, labels_name)
    plt.tight_layout()
    plt.show()  # Plot latest log


def save_all(labels_name, latest_log=''):
    if latest_log == '':
        latest_log, _ = get_latest_log(results_dir)
        latest_log = latest_log[1:]

    tra_acc_list, val_acc_list, loss_list, cm = load_history(
        latest_log=latest_log)

    save_acc(tra_acc_list, val_acc_list, results_dir + '/' + latest_log)
    save_loss(loss_list, results_dir + '/' + latest_log)
    save_confusion_matrix(
        cm, labels_name, save_path=results_dir + '/' + latest_log)
    print('Re-saved.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot')
    parser.add_argument('--log', type=str, default='')
    args = parser.parse_args()
    # Default will re-save latest log
    classes = ['PearlRiver', 'YoungChang', 'Steinway-T',
               'Hsinghai', 'Kawai', 'Steinway', 'Kawai-G', 'Yamaha']
    if classes is None:
        ds = load_dataset("CCOM/pianos_mel")
        classes = ds['test'].features['label'].names

    save_all(labels_name=classes, latest_log=args.log)
