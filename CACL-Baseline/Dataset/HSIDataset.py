import torch
from torch.utils.data import Dataset
import numpy as np
from utils import utils


class HSIDataset(object):
    def __init__(self, patchsz):
        super().__init__()
        self.patchsz = patchsz

    def Normalize(self, data):
        h, w, c = data.shape
        data = data.reshape((h * w, c))
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0)
        data = data.reshape((h, w, c))
        return data

    def addMirror(self, data):
        dx = self.patchsz // 2
        h, w, bands = data.shape
        mirror = None
        if dx != 0:
            mirror = np.zeros((h + 2 * dx, w + 2 * dx, bands))  
            mirror[dx:-dx, dx:-dx, :] = data 
            for i in range(dx):  
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
        return mirror

    def generate(self, data, label):
        indices = list(zip(*np.nonzero(label)))
        sample = np.zeros((len(indices), self.patchsz, self.patchsz, data.shape[-1]), dtype=np.float32)
        for i, (x, y) in enumerate(indices):
            sample[i] = data[x:x + self.patchsz, y:y + self.patchsz]
        indices = tuple(zip(*indices))
        label = label[indices] - 1
        return sample, label


class LabelDataset(Dataset, HSIDataset):
    def __init__(self, data, label, patchsz=7, is_train=False):
        super().__init__(patchsz)
        if data.dtype != np.float32: data = data.astype(np.float32)
        if label.dtype != np.int32: label = label.astype(np.int32)

        data = self.addMirror(data)
        data = 2 * self.Normalize(data) - 1
        self.data, self.label = self.generate(data, label)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return {"x_lb": torch.tensor(self.data[index], dtype=torch.float32),
                "y_lb": torch.tensor(self.label[index], dtype=torch.long)}

class UnlabelDataset(HSIDataset):
    def __init__(self, data, label, patchsz):
        super().__init__(patchsz)
        if data.dtype != np.float32: data = data.astype(np.float32)
        if label.dtype != np.int32: label = label.astype(np.int32)

        data = self.addMirror(data)
        self.data = 2 * self.Normalize(data) - 1
        self.indices = list(zip(*np.nonzero(label)))
        self.len = len(self.indices)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x_coor, y_coor = self.indices[index]
        x = self.data[x_coor:x_coor + self.patchsz, y_coor:y_coor + self.patchsz]
        return {"x_ulb": torch.tensor(x, dtype=torch.float32)}
    

class LabelDataset_dynamic(HSIDataset):
    def __init__(self, data, label, patchsz=7, is_train=False):
        super().__init__(patchsz)
        if data.dtype != np.float32: data = data.astype(np.float32)
        if label.dtype != np.int32: label = label.astype(np.int32)

        data = self.addMirror(data)
        self.data = 2 * self.Normalize(data) - 1
        self.indices = list(zip(*np.nonzero(label)))
        self.len = len(self.indices)
        self.label = label
        self.is_train = is_train

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x_coor, y_coor = self.indices[index]
        x = self.data[x_coor:x_coor + self.patchsz, y_coor:y_coor + self.patchsz]
        y = self.label[x_coor, y_coor] - 1
        return {"x_lb": torch.tensor(x, dtype=torch.float32),
                "y_lb": torch.tensor(y, dtype=torch.long)}


class AllPixelDataset(HSIDataset):
    def __init__(self, data, os_label, patchsz):
        super().__init__(patchsz)
        if data.dtype != np.float32: data = data.astype(np.float32)
        if os_label.dtype != np.int32: os_label = os_label.astype(np.int32)
        self.h, self.w = data.shape[0], data.shape[1]
        data = self.addMirror(data)
        self.data = 2 * self.Normalize(data) - 1
        os_label = os_label.reshape(self.h * self.w)
        indices = np.nonzero(os_label)[0]
        self.indices = indices
        self.all_os_label = torch.tensor(os_label, dtype=torch.long)
        self.label = torch.tensor(os_label[indices], dtype=torch.long)
        self.label -= 1
        self.len = self.h * self.w

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x_coor, y_coor = index // self.w, index % self.w
        x = self.data[x_coor:x_coor + self.patchsz, y_coor:y_coor + self.patchsz]
        return {"x_lb": torch.tensor(x, dtype=torch.float32)}


class DatasetInfo(object):
    info = {
        'salinas': {
            'data_file_name': 'salinas_corrected.mat',
            'data_key': 'salinas_corrected',
            'gt_file_name': 'salinas_gt.mat',
            'label_key': 'salinas_gt',
            'class_num': 16,
    },
        'Indian':{
            'data_file_name': 'Indian_pines_corrected.mat',
            'data_key': 'indian_pines_corrected',
            'gt_file_name': 'Indian_gt.mat',
            'label_key': 'indian_pines_gt',
            'class_num': 16,
    },
        'paviaU':{
            'gt_file_name': 'paviaU_gt.mat',
            'label_key': 'paviaU_gt',
            'data_file_name': 'paviaU.mat',
            'data_key': 'paviaU',
            'class_num': 9,
        }
    }
