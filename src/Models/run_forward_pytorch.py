import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
