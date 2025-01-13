import torch
import torch.nn as nn

class FECNNClassifier(nn.Module):
    def __init__(_self, _in_features:int, _num_classes:int, _accel_device:str="cpu", _dtype=torch.float32):
        super().__init__()

        _self.conv1 = nn.Conv1d(1, 4, 3, padding=1, device=_accel_device, dtype=_dtype)
        _self.conv2 = nn.Conv1d(4, 8, 3, padding=1)
        _self.out_layer = nn.Linear(_in_features * 8, _num_classes, device=_accel_device, dtype=_dtype)
        _self.batch_norm1 = nn.BatchNorm1d(4, device=_accel_device, dtype=_dtype)
        _self.batch_norm2 = nn.BatchNorm1d(8, device=_accel_device, dtype=_dtype)
        _self.leaky_relu = nn.LeakyReLU()
        _self.flat_layer = nn.Flatten()

    def forward(_self, x:torch.Tensor) -> torch.Tensor:
        x = _self.conv1(x)
        x = _self.batch_norm1(x)
        x = _self.leaky_relu(x)
        x = _self.conv2(x)
        x = _self.batch_norm2(x)
        x = _self.leaky_relu(x)
        x = _self.flat_layer(x)

        return _self.out_layer(x)
