import torch
import torch.nn as nn

class FEMLPClassifier(nn.Module):
    def __init__(_self, _width:int, _depth:int, _in_features:int, _num_classes:int, _accel_device:str="cpu", _dtype=torch.float32):
        super().__init__()

        _self.in_layer = nn.Linear(_in_features, _width, device=_accel_device, dtype=_dtype)
        _self.leaky_relu = nn.LeakyReLU()
        _self.linears = nn.ModuleList()

        for i in range(len(_depth)):
            _self.linears.append(nn.Linear(_width, _width, device=_accel_device, dtype=_dtype))

        _self.out_layer = nn.Linear(_width, _num_classes)

    def forward(_self, x:torch.Tensor) -> torch.Tensor:
        x = _self.in_layer(x)
        x = _self.leaky_relu(x)

        for layer in _self.linears:
            x = layer(x)
            x = _self.leaky_relu(x)

        return _self.out_layer(x)
