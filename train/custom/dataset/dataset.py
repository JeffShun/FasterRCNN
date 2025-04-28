"""data loader."""

import numpy as np
from torch.utils import data
import cv2

class MyDataset(data.Dataset):
    def __init__(
            self,
            dst_list_file,
            transforms
        ):
        self.data_lst = self._load_files(dst_list_file)
        self._transforms = transforms

    def _load_files(self, file):
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, idx):
        source_data = self._load_source_data(self.data_lst[idx])
        return source_data

    def __len__(self):
        return len(self.data_lst)

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        img = data['img']
        target = data['label']
        # transform前，数据必须转化为[C,H,W]的形状
        img = np.transpose(img, (2, 0, 1))
        img = img[:,:].astype(np.float32)
        target = target.astype(np.float32)
        if self._transforms:
            img, target = self._transforms(img, target)
        return img, target

