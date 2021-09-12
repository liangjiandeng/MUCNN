import torch.utils.data as data
import torch
import numpy as np
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path)

        self.gt = dataset.get("gt")  # NxCxHxW
        self.ms = dataset.get("ms")
        self.lms = dataset.get("lms")
        self.pan = dataset.get("pan")
        # print(self.ms.shape)
        # print(self.pan.shape)
        # print(dataset)
        # input()

    #####必要函数
    def __getitem__(self, index):
        return torch.from_numpy(self.gt[index, :, :, :]/ 2047).float(), \
               torch.from_numpy(self.lms[index, :, :, :]/ 2047).float(), \
               torch.from_numpy(self.ms[index, :, :, :]/ 2047).float(), \
               torch.from_numpy(self.pan[index, :, :, :]/ 2047).float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]
