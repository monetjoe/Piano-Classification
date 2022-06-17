
import csv
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from alexnet import AlexNet
from data import prepare_data, create_dir
from focaloss import FocalLoss
from plotter import save_acc, save_loss


def eval_model_train(model, trainLoader, device, tra_acc_list):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of trainloader: %d %%' % (100 * correct / total))
    tra_acc_list.append(100 * correct / total)


def eval_model_validation(model, validationLoader, device, val_acc_list):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validationLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of validationloader: %d %%' % (100 * correct / total))
    val_acc_list.append(100 * correct / total)


def eval_model_test(model, testLoader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of test: %d %%' % (100 * correct / total))


def time_stamp():
    now = int(round(time.time()*1000))
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(now / 1000))


def save_history(tra_acc_list, val_acc_list, loss_list):

    log_dir = './logs/history_' + time_stamp()
    create_dir(log_dir)

    acc_len = len(tra_acc_list)
    with open(log_dir + "/acc.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tra_acc_list", "val_acc_list"])
        for i in range(acc_len):
            writer.writerow([tra_acc_list[i], val_acc_list[i]])

    loss_len = len(loss_list)
    with open(log_dir + "/loss.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["loss_list"])
        for i in range(loss_len):
            writer.writerow([loss_list[i]])

    save_acc(tra_acc_list, val_acc_list, log_dir + "/acc.png")
    save_loss(loss_list, log_dir + "/loss.png")

    return log_dir


def train(epoch_num=40, iteration=10, lr=0.001):
    print('Start training...')
    tra_acc_list, val_acc_list, loss_list = [], [], []

    # print(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    trainLoader, validLoader, testLoader = prepare_data()

    # init model
    model = AlexNet()

    #optimizer and loss
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(class_num=7)
    optimizer = optim.SGD(model.classifier.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=lr, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # gpu
    torch.cuda.empty_cache()
    model = model.cuda()
    criterion = criterion.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # train process
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        epoch_str = f' Epoch {epoch + 1}/{epoch_num} '
        print(f'{epoch_str:-^40s}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            # get the inputs
            inputs, labels = data[0].cuda(), data[1].cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % iteration == iteration - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / iteration))
                loss_list.append(running_loss / iteration)
            running_loss = 0.0

        eval_model_train(model, trainLoader, device, tra_acc_list)
        eval_model_validation(model, validLoader, device, val_acc_list)
        scheduler.step(loss.item())

    print('Finished Training')
    eval_model_test(model, testLoader, device)
    log_dir = save_history(tra_acc_list, val_acc_list, loss_list)
    torch.save(model, log_dir + '/save.pt')


if __name__ == "__main__":

    train(epoch_num=40)
