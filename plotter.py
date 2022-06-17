import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from train import train
from scipy import signal as ss
from data import create_dir


def show_point(max_id, list):
    show_max = '(' + str(max_id + 1) + ', ' + str(round(list[max_id], 2)) + ')'
    plt.annotate(show_max, xytext=(
        max_id + 1, list[max_id]), xy=(max_id + 1, list[max_id]), fontsize=6)


def smooth(y):
    return ss.savgol_filter(y, 95, 3)


def plot_acc(tra_acc_list, val_acc_list):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    max1 = np.argmax(y1)
    max2 = np.argmax(y2)

    plt.subplot(121)
    plt.title('Accuracy of training and validation')
    plt.xlabel('Epoch')
    plt.ylabel('Acc(%)')
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

    plt.title('Accuracy of training and validation')
    plt.xlabel('Epoch')
    plt.ylabel('Acc(%)')
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")
    plt.plot(1+max1, y1[max1], 'r-o')
    plt.plot(1+max2, y2[max2], 'r-o')
    show_point(max1, y1)
    show_point(max2, y2)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def save_loss(loss_list, save_path):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.title('Loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(x_loss, smooth(loss_list))
    plt.savefig(save_path)
    plt.close()


def plot_loss(loss_list):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.subplot(122)
    plt.title('Loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(x_loss, smooth(loss_list))


def get_latest_log(path):
    lists = os.listdir(path)
    lists.sort(key=lambda x: os.path.getctime((path + "\\" + x)))
    return '/' + lists[-1]


def valid_path(log_path, latest_log):
    if latest_log == '':
        return False

    dir_path = log_path + '/' + latest_log
    return os.path.exists(dir_path)


def load_history(log_dir='./logs', latest_log=''):

    create_dir(log_dir)

    if len(os.listdir(log_dir)) == 0:
        print('Please finish training first.')
        exit()

    if valid_path(log_dir, latest_log):
        latest_log = '/' + latest_log
    else:
        latest_log = get_latest_log(log_dir)

    latest_acc = log_dir + latest_log + '/acc.csv'
    latest_loss = log_dir + latest_log + '/loss.csv'
    acc_list = pd.read_csv(latest_acc)
    tra_acc_list = acc_list['tra_acc_list'].tolist()
    val_acc_list = acc_list['val_acc_list'].tolist()
    loss_list = pd.read_csv(latest_loss)['loss_list'].tolist()
    return tra_acc_list, val_acc_list, loss_list


def plot_all(latest_log=''):
    tra_acc_list, val_acc_list, loss_list = load_history(latest_log=latest_log)

    plt.figure(figsize=(12, 4.5))
    plot_acc(tra_acc_list, val_acc_list)
    plot_loss(loss_list)
    plt.show()  # Plot latest log


def save_all(latest_log):
    tra_acc_list, val_acc_list, loss_list = load_history(latest_log=latest_log)
    save_acc(tra_acc_list, val_acc_list, './logs/' + latest_log + '/acc.png')
    save_loss(loss_list, './logs/' + latest_log + '/loss.png')
    print('Re-saved.')


if __name__ == "__main__":
    plot_all()
    # plot_all(latest_log='history_2022-06-16_06-20-26')
    # save_all('history_2022-06-16_06-20-26')
