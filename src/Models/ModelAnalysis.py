import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skmetrics

from torch.utils.data import DataLoader

def run_train_forward_pytorch(_model:nn.Module,
                              _train_dataloader:DataLoader,
                              _val_dataloader:DataLoader,
                              _num_epochs:int,
                              _hyperparameters:dict,
                              _accel_device:str="cpu",
                              _dtype=torch.float32) -> dict:
    train_losses = []

    val_losses = []
    
    train_accs = []

    val_accs = []

    optimizer = None

    if _hyperparameters["optim_name"].__eq__("adam"):
        optimizer = optim.Adam(_model.parameters(), _hyperparameters["lr"], _hyperparameters["betas"], _hyperparameters["eps"], _hyperparameters["weight_decay"])
    elif _hyperparameters["optim_name"].__eq__("rmsprop"):
        optimizer = optim.RMSprop(_model.parameters(), _hyperparameters["lr"], _hyperparameters["alpha"], _hyperparameters["eps"], _hyperparameters["weight_decay"])
    elif _hyperparameters["optim_name"].__eq__("momentum"):
        optimizer = optim.SGD(_model.parameters(),
                              _hyperparameters["lr"],
                              _hyperparameters["momentum"],
                              weight_decay=_hyperparameters["eps"],
                              nesterov=_hyperparameters["nestorov"])
    else:
        optimizer = optim.SGD(_model.parameters(), _hyperparameters["lr"], weight_decay=_hyperparameters["weight_decay"])

    num_train_batches = len(_train_dataloader)
    
    num_val_batches = len(_val_dataloader)

    for epoch in range(_num_epochs):
        _model.train()

        total_train_loss = 0.0

        total_train_acc = 0.0

        for train_data, train_label in _train_dataloader:
            train_data, train_label = train_data.to(_accel_device), train_label.to(device=_accel_device, dtype=torch.int64)

            optimizer.zero_grad()

            train_output = _model(train_data)
            train_loss = F.cross_entropy(train_output, train_label)
            train_loss.backward()

            optimizer.step()

            total_train_loss += train_loss.item()

            train_predictions = torch.argmax(F.softmax(train_output, 1, _dtype), 1).to(torch.int64).view(-1)
            total_train_acc += ((train_predictions == train_label).sum() / len(train_label)).item() * 100

        train_losses.append(total_train_loss / num_train_batches)
        
        train_accs.append(total_train_acc / num_train_batches)

        _model.eval()

        total_val_loss = 0.0

        total_val_acc = 0.0

        for val_data, val_label in _val_dataloader:
            val_data, val_label = val_data.to(_accel_device), val_label.to(device=_accel_device, dtype=torch.int64)

            val_output = _model(val_data)
            val_loss = F.cross_entropy(val_output, val_label)
            total_val_loss += val_loss.item()

            val_predictions = torch.argmax(F.softmax(val_output, 1, _dtype), 1).to(torch.int64).view(-1)
            total_val_acc += ((val_predictions == val_label).sum() / len(val_label)).item() * 100

        val_losses.append(total_val_loss / num_val_batches)
        
        val_accs.append(total_val_acc / num_val_batches)

        print(f"[INFO] Epoch {epoch + 1}/{_num_epochs}: Train/Validation loss = {train_losses[-1]:.4f}/{val_losses[-1]:.4f}, train/validation accuracy = {train_accs[-1]:.2f}%/{val_accs[-1]:.2f}%\n")

        if len(val_losses) >= 2 and  val_losses[-1] >= val_losses[-2]:
            break

    return { "train_losses": train_losses,"train_accs": train_accs,"val_losses":val_losses,"val_accs":val_accs }


def run_predict_forward(_model:nn.Module, _dataloader:DataLoader, _accel_device:str="cpu", _dtype=torch.int64):
    predictions = []

    labels = []

    for batch_data, batch_label in _dataloader:
        batch_data, batch_label = batch_data.to(_accel_device), batch_label.to(device=_accel_device, dtype=_dtype)

        output = _model(batch_data)
        batch_predictions = torch.argmax(F.softmax(output, 1, _dtype), 1).to(torch.int64).view(-1)
        for prediction in batch_predictions:
            predictions.append(prediction.item())

        for label in batch_label:
            labels.append(label.item())

    return predictions, labels


def show_analysis_charts(_losses:list, _accuracies:list, _val_losses:list=None, _val_accuracies:list=None, _mode_name:str="Training", _optimizer_name:str="SGD"):
    plt.style.use('seaborn-v0_8-dark')

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(16,6))

    loss_ax.plot(np.arange(1, len(_losses) + 1), _losses, label=_mode_name)

    if _val_losses != None:
        loss_ax.plot(np.arange(1, len(_val_losses) + 1), _val_losses, label="Validation")

    loss_ax.set_xlabel(f"Epoch Iteration")
    loss_ax.set_xticks(np.arange(1, len(_losses) + 1))
    loss_ax.set_ylabel("Categorical Cross Entropy (CCE) loss")
    loss_ax.set_title(f"{_mode_name} Epoch vs. Loss ({_optimizer_name})")
    loss_ax.legend()
    loss_ax.grid()

    acc_ax.plot(np.arange(1, len(_accuracies) + 1), _accuracies, label=_mode_name)

    if _val_accuracies != None:
        acc_ax.plot(np.arange(1, len(_val_accuracies) + 1), _val_accuracies, label="Validation")

    acc_ax.set_xlabel(f"Epoch Iteration")
    acc_ax.set_xticks(np.arange(1, len(_accuracies) + 1))
    acc_ax.set_ylabel("Accuracy (in %)")
    acc_ax.set_yticks(np.arange(0, 110, 10))
    acc_ax.set_title(f"{_mode_name} Epoch vs. Accuracy ({_optimizer_name})")
    acc_ax.legend()
    acc_ax.grid()

    plt.show()


def show_metrics(_all_preds:list, _all_labels:list, label_names:list, _val_preds:list=None, _val_labels:list=None, _mode_name:str="Training"):
    print(f"[INFO] {_mode_name} Classification Report:")
    print(skmetrics.classification_report(_all_labels, _all_preds, target_names=label_names, zero_division=0.0))
    
    if _val_preds != None and _val_labels != None:
        print(f"[INFO] Validation Classification Report:")
        print(skmetrics.classification_report(_val_labels, _val_preds, zero_division=0.0))

    print(f"[INFO] {_mode_name} Confusion Matrix:")
    matrix = skmetrics.ConfusionMatrixDisplay(skmetrics.confusion_matrix(_all_labels, _all_preds), display_labels=label_names)
    matrix.plot()

    plt.show()

    if _val_preds != None and _val_labels != None:
        print(f"[INFO] Validation Confusion Matrix:")
        matrix = skmetrics.ConfusionMatrixDisplay(skmetrics.confusion_matrix(_val_labels, _val_preds))
        matrix.plot()

        plt.show()
