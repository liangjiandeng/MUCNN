import torch
import torchvision
import torch.nn as nn
import scipy.io as sio
import numpy as np
import torch.utils.data as data
import h5py
from lib.utils import *
import torchvision.transforms as transforms


class Datasets(data.Dataset):

    def __init__(self, path, device):
        super().__init__()
        dataset = np.array(h5py.File(path))

        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.
        self.ms = torch.from_numpy(ms1)  # NxCxHxW:

        self.ms = dataset.get("ms")
        self.lms = dataset.get("lms")
        self.pan = dataset.get("pan")
        # self.data = sio.loadmat(path)
        # if self.data['gt'].ndim == 4:
        #     self.size = self.data['gt'].shape[0]
        # else:
        #     self.size = 1
        # if self.data['gt'].ndim == 3:
        #     gt = np.transpose(self.data['gt'], (2, 0, 1))
        #     self.gt = torch.from_numpy(
        #         np.array(gt, dtype=np.float32) / 2047
        #     ).reshape(1, gt.shape[0], gt.shape[1], gt.shape[2])
        # else:
        #     self.gt = torch.from_numpy(
        #         np.transpose(
        #             np.array(self.data['gt'][...], dtype=np.float32), (0, 3, 1, 2)
        #         ) / 2047
        #     )
        # if self.data['ms'].ndim == 3:
        #     ms = np.transpose(self.data['ms'], (2, 0, 1))
        #     self.ms = torch.from_numpy(
        #         np.array(ms, dtype=np.float32) / 2047
        #     ).reshape(1, ms.shape[0], ms.shape[1], ms.shape[2])
        # else:
        #     self.ms = torch.from_numpy(
        #         np.transpose(
        #             np.array(self.data['ms'][...], dtype=np.float32), (0, 3, 1, 2)
        #         ) / 2047
        #     )
        # if self.data['pan'].ndim == 2:
        #     pan = self.data['pan']
        #     self.pan = torch.from_numpy(
        #         np.array(pan, dtype=np.float32) / 2047
        #     ).reshape(1, 1, gt.shape[1], gt.shape[2])
        # elif self.data['pan'].ndim == 3:
        #     self.size = self.gt.shape[0]
        #     pan = self.data['pan']
        #     self.pan = torch.from_numpy(
        #         np.array(pan, dtype=np.float32) / 2047
        #     ).reshape(pan.shape[0], 1, pan.shape[1], pan.shape[2])
        # else:
        #     self.pan = torch.from_numpy(
        #         np.transpose(
        #             np.array(self.data['pan'][...], dtype=np.float32), (0, 3, 1, 2)
        #         ) / 2047
        #     )
        # # self.size = self.gt.shape[0]
        # pan = self.data['pan']
        # self.pan = torch.from_numpy(
        #     np.array(pan, dtype=np.float32) / 2047
        # ).reshape(pan.shape[0], 1, pan.shape[1], pan.shape[2])
        # self.ms = torch.from_numpy(
        #     np.transpose(
        #         np.array(self.data['ms'][...], dtype=np.float32), (0, 3, 1, 2)
        #     ) / 2047
        # )
        self.size = self.gt.shape[0]
        self.gt1 = torch.nn.functional.interpolate(self.gt, scale_factor=0.5)
        self.gt2 = torch.nn.functional.interpolate(self.gt, scale_factor=0.25)

    def __getitem__(self, index):
        return torch.from_numpy(self.gt[index, :, :, :]/ 2047).float(), \
               torch.from_numpy(self.lms[index, :, :, :]/ 2047).float(), \
               torch.from_numpy(self.ms[index, :, :, :]/ 2047).float(), \
               torch.from_numpy(self.pan[index, :, :, :]/ 2047).float()



def tensor_pil(tensor):
    return torchvision.transforms.ToPILImage(tensor)


if __name__ == '__main__':
    train_data = Datasets('../data/train.mat', 'cuda')
