import csv
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from datetime import datetime
from model import Net, FocalLoss, models
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from plot import save_acc, save_loss, save_confusion_matrix
from data import prepare_data, load_data
from utils import *


def eval_model_train(model, trainLoader, tra_acc_list: list):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in tqdm(trainLoader, desc="Batch evaluation on trainset..."):
            inputs, labels = toCUDA(data["mel"]), toCUDA(data["label"])
            outputs: torch.Tensor = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print(f"\nTraining acc   : {str(round(acc, 2))}%")
    tra_acc_list.append(acc)


def eval_model_valid(
    model: nn.Module,
    validationLoader,
    val_acc_list: list,
    log_dir: str,
    best_acc: float,
):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in tqdm(validationLoader, desc="Batch evaluation on validset..."):
            inputs, labels = toCUDA(data["mel"]), toCUDA(data["label"])
            outputs: torch.Tensor = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print(f"\nValidation acc : {str(round(acc, 2))}%")
    val_acc_list.append(acc)

    if acc > best_acc:
        torch.save(model.state_dict(), f"{log_dir}/save.pt")
        print("Model saved.")
        return acc

    else:
        return best_acc


def eval_model_test(log_dir: str, backbone_ver: str, testLoader, classes):
    model = Net(len(classes), m_ver=backbone_ver, saved_model_path=f"{log_dir}/save.pt")
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in tqdm(testLoader, desc="Batch evaluation on testset..."):
            inputs, labels = toCUDA(data["mel"]), toCUDA(data["label"])
            outputs: torch.Tensor = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    cm = confusion_matrix(y_true, y_pred, normalize="all")

    return report, cm


def save_log(
    start_time: datetime,
    finish_time: datetime,
    cls_report,
    cm,
    log_dir,
    classes,
):
    logs = f"""
Backbone     : {args.model}
Start time   : {start_time}"
Finish time  : {finish_time}"
Time cost    : {(finish_time - start_time).seconds}s"
Full finetune: {args.fullfinetune}"
Focal loss   : {args.fl}"""

    with open(f"{log_dir}/result.log", "w", encoding="utf-8") as f:
        f.write(cls_report + "\n" + logs + "\n")

    # save confusion_matrix
    np.savetxt(f"{log_dir}/mat.csv", cm, delimiter=",")
    save_confusion_matrix(cm, classes, log_dir)

    print(cls_report)
    print("Confusion matrix :")
    print(str(cm.round(3)) + "\n")
    print(logs)


def save_history(
    log_dir,
    tra_acc_list,
    val_acc_list,
    loss_list,
    lr_list,
    cls_report,
    cm,
    start_time,
    finish_time,
    classes,
):
    acc_len = len(tra_acc_list)
    with open(f"{log_dir}/acc.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tra_acc_list", "val_acc_list", "lr_list"])
        for i in range(acc_len):
            writer.writerow([tra_acc_list[i], val_acc_list[i], lr_list[i]])

    loss_len = len(loss_list)
    with open(f"{log_dir}/loss.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["loss_list"])
        for i in range(loss_len):
            writer.writerow([loss_list[i]])

    save_acc(tra_acc_list, val_acc_list, log_dir)
    save_loss(loss_list, log_dir)
    save_log(start_time, finish_time, cls_report, cm, log_dir, classes)


def train(backbone_ver="squeezenet1_1", epoch_num=40, iteration=10, lr=0.001):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tra_acc_list, val_acc_list, loss_list, lr_list = [], [], [], []

    # load data
    ds, classes, num_samples, use_hf = prepare_data(args.fl)
    cls_num = len(classes)

    # init model
    model = Net(cls_num, m_ver=backbone_ver, full_finetune=args.fullfinetune)
    input_size = model._get_insize()
    traLoader, valLoader, tesLoader = load_data(ds, input_size, use_hf)

    # optimizer and loss
    criterion = FocalLoss(num_samples) if args.fl else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
        verbose=True,
        threshold=lr,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-08,
    )

    # gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        criterion = criterion.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # train process
    start_time = datetime.now()
    log_dir = f"{LOGS_DIR}/{args.model}__{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    create_dir(log_dir)
    print(f"Start training {args.model} at {start_time}...")
    # loop over the dataset multiple times
    for epoch in range(epoch_num):
        epoch_str = f" Epoch {epoch + 1}/{epoch_num} "
        lr_str = optimizer.param_groups[0]["lr"]
        lr_list.append(lr_str)
        print(f"{epoch_str:-^40s}")
        print(f"Learning rate: {lr_str}")
        running_loss = 0.0
        best_eval_acc = 0.0
        with tqdm(total=len(traLoader), unit="batch") as pbar:
            for i, data in enumerate(traLoader, 0):
                # get the inputs
                inputs, labels = toCUDA(data["mel"]), toCUDA(data["label"])
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model.forward(inputs)
                loss: torch.Tensor = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                # print every 2000 mini-batches
                if i % iteration == iteration - 1:
                    pbar.set_description(
                        "epoch=%d/%d, lr=%.4f, loss=%.4f"
                        % (
                            epoch + 1,
                            epoch_num,
                            lr,
                            running_loss / iteration,
                        )
                    )
                    loss_list.append(running_loss / iteration)

                running_loss = 0.0
                pbar.update(1)

            eval_model_train(model, traLoader, tra_acc_list)
            best_eval_acc = eval_model_valid(
                model,
                valLoader,
                val_acc_list,
                log_dir,
                best_eval_acc,
            )
            scheduler.step(loss.item())

    finish_time = datetime.now()
    cls_report, cm = eval_model_test(log_dir, backbone_ver, tesLoader, classes)
    save_history(
        log_dir,
        tra_acc_list,
        val_acc_list,
        loss_list,
        lr_list,
        cls_report,
        cm,
        start_time,
        finish_time,
        classes,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--model", type=str, default="squeezenet1_1")
    parser.add_argument("--fl", type=bool, default=False)
    parser.add_argument("--fullfinetune", type=bool, default=False)
    args = parser.parse_args()

    train(backbone_ver=args.model, epoch_num=2)  # 2 for test
