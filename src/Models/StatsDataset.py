import torch
from torch.utils.data import Dataset
from StatsTableSingleton import StatsTableSingleton

class StatsDataset(Dataset):
    def __init__(_self, _accel_device:str="cpu", _dtype=torch.float32, _data_transform=None, target_transform=None):
        stats_dataframe = StatsTableSingleton().get_all_stats()
        _self.stats_data = torch.Tensor(stats_dataframe.drop("character", axis=1).to_numpy()).to(device=_accel_device, dtype=_dtype)
        _self.transform = _data_transform
        _self.target_transform = target_transform

    def __len__(_self):
        return len(_self.stats_data)

    def __getitem__(_self, _index):
        data = _self.stats_data[_index, :-1]

        label = _self.stats_data[_index, -1]

        if _self.transform:
            data = _self.transform(data)

        if _self.target_transform:
            label = _self.target_transform(label)

        return data, label
